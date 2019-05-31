import os,io,sys
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import numpy as np
import cv2
import timeit,csv,collections

from google.cloud import videointelligence_v1p2beta1 as videointelligence
from difflib import SequenceMatcher

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "VideoTextDetection-91e1475e2aea.json"


'''get a list of black screen time'''
# detect the black screen of the whole video
# TODO: faster 1. 2-5 frames/sec instead of 25
# TODO 2.other option: moviepy.video.VideoClip.ColorClip(size, color=None, ismask=False, duration=None, col=None)
def getAllBlackFrame(path):
    vid_frame = cv2.VideoCapture(path)
    length = vid_frame.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = vid_frame.get(cv2.CAP_PROP_FPS)
    filelenth = int(length/fps)

    # check whole video
    num_frames = int(length)
    intensity = []
    blackscreens = []

    for i in range(num_frames):
        success, image = vid_frame.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if success:
            total = np.sum(np.sum(image, axis=0))
            intensity.append(total / (image.shape[0] * image.shape[1]))
            if total == 0:
                # gives the intro end time in the format x min: y sec
                timestring = str(int(i / fps / 60)) + " min: " + str(int(i / fps % 60)) + "sec"
                timestamp = int(i / fps)
                if timestamp not in blackscreens:
                    blackscreens.append(timestamp)  # gives the intro end time in the format x min: y sec
                    print(timestring + ' {}'.format(timestamp))

    return filelenth,blackscreens


# go through the black screen and get the detection start timestamp
# black screens close to each other are seen as one
''' input: list of black screen
    output:list of start checking points based on black screen'''


def get_checkpoints(BLFrame):

    check_points = []
    for idx, c_point in enumerate(BLFrame):
        if idx > 0:
            p_point = BLFrame[idx - 1]
            if (c_point - p_point) > 1:
                check_points.append(p_point)

    check_points.append(BLFrame[-1]) #add the last black screen time
    return check_points



# text detection with Google video intelligence api
# output: text detected in the clip, list of Data [Text, Time, Confi]
def textDetection(path):

    with io.open(path, 'rb') as file:
        input_content = file.read()
    ResultList = []

    video_client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.enums.Feature.TEXT_DETECTION]
    video_context = videointelligence.types.VideoContext()

    operation = video_client.annotate_video(
        input_content=input_content,  # the bytes of the video file
        features=features,
        video_context=video_context)

    print('\nProcessing video for text detection. timeout is 300s')
    result = operation.result(timeout=300)

    # TODO: batch process is possible
    # The first result is retrieved because a single video was processed.
    annotation_result = result.annotation_results[0]

    #extract results from data structure annotation_result
    for text_annotation in annotation_result.text_annotations:
        text_segment = text_annotation.segments[0]
        start_time = text_segment.segment.start_time_offset

        # get rid of results which confidence of text detection is less than 0.5
        if text_segment.confidence < 0.5:
            continue

        # Data item: [Text, Time, Confi]
        # TODO: checking all utf - 8 text
        d = Data(text_annotation.text.encode('utf-8'), start_time.seconds + start_time.nanos * 1e-9, text_segment.confidence)
        ResultList.append(d)

    # ResultList.sort(key=setSortedKey)
    ResultList.sort(key=lambda x: x[1])
    return ResultList


# set the sort function: result will be sorted by time
# used by textDetection func
# def setSortedKey(p):
#     return p[1]


def getCast(video_name):
    casts = []
    with open('cast.csv', 'rU') as csvfile:
        names = list(csv.DictReader(csvfile))
    for n in names:
        casts.append(n[video_name])

    return casts


# get a relevance of text detection with metadata
# compare the list of names with text detection

def getSimilarity(text, cast):
    cast = map(lambda x: x.lower(), cast)
    text = map(lambda x: x.lower(), text)
    right = 0.0;

    for tt in text:
        if tt in cast:
            simi = 1
            print('cast:{}, detect:{}, simi:{}'.format(tt, tt, simi))
        else:
            simi = 0
            for s in cast: #similarity between two strings if not 100% match
                temps = SequenceMatcher(None, tt, s).ratio()
                simi = temps  if temps > simi else simi
            print('cast:{}, simi:{}'.format(tt, SequenceMatcher(None, tt, s).ratio()))
        right += simi

    return right/len(text)


def checkTimeofCreditScene(text_on_screen,video_start):
    if text_on_screen:
        text_in_sec =[]
        current_sec = 0;
        for idx, text_det in enumerate(text_on_screen):
            if idx == 0 :
                text_in_sec.append(text_det.Text)
                current_sec = text_det.Time
            # new text det time is different with current checking second
            # get the similarity of current list
            elif text_det.Time != current_sec:
                real_time = video_start + current_sec
                relevance = getSimilarity(text_in_sec,cast_names)

                split_time_string = str(int(real_time / 60)) + " min: " + str(real_time % 60) + "sec"
                print('checking from : {} possibility:{}'.format(split_time_string, relevance))
                # if the relevance is high enough, print out the time
                if relevance > 0.7:
                    split_time_string = str(int(real_time / 60)) + " min: " + str(real_time % 60) + "sec"
                    print('ending part starts at : {} possibility:{}'.format(split_time_string,relevance))
                    return split_time_string
                # if the relevance is low, start a new list of next second
                else:
                    text_in_sec = []
                    text_in_sec.append(text_det.Text)
                    current_sec = text_det.Time
                    continue
            else:
                text_in_sec.append(text_det.Text)




"""Main func start"""

# fileName = 'Bullets7Last5min.mp4'
fileName =  "videos/Bullets7.mp4"

filelenth = 900  #default 5 mins
clipLen = 10 # size of sub clips from the checking points

# get the metadata of the movie: names of movie/TV directors, actors...
video_name = 'Bullets'
cast_names = getCast(video_name)

Data = collections.namedtuple('Data', ['Text', 'Time', 'Confi'])


''' find black screen as checking points, or start from beginning'''
# startRunning = timeit.default_timer()
# filelenth, blackFrames = getAllBlackFrame(path=fileName)
# if not blackFrames: #start detecting text from the beginning if there is no black screen
#     checkPoints = range(0,filelenth,clipLen)
# else:
#     checkPoints = get_checkpoints(blackFrames)
# print('get black screen running time: ',timeit.default_timer()- startRunning)


# checkPoints = [0, 5, 78, 160, 165, 679, 890, 894]
checkPoints = [679, 890, 894]

''' for each checking point, detect text on the screen'''
for ind,p in enumerate(checkPoints):

    # avoid checking overlapped clip
    if ind > 0:
        if p - checkPoints[ind - 1] < 10:
            p = checkPoints[ind - 1] + clipLen
    if filelenth - p < clipLen:
        clipLen = filelenth - p

    p -= 1 if p != 0 else p #1 sec before black end

    # generate a video clip
    new_file = "temp.mp4"
    if clipLen > 0:
        ffmpeg_extract_subclip(fileName, p, p + clipLen, targetname=new_file)
    print ('checking clip from ',str(p),str(clipLen))


    # get text results in clip
    # Out:  Data item list  - Text, Time, Confi
    startRunning = timeit.default_timer()
    textItem_detected = textDetection(new_file)
    print('run text detection time : ', timeit.default_timer() - startRunning)

    startRunning = timeit.default_timer()
    ending_time = checkTimeofCreditScene(textItem_detected, p)
    print('get similarity time : ', timeit.default_timer() - startRunning)


    if ending_time != None:
        print(ending_time)
        break

















