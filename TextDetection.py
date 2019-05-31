# key AIzaSyAyB1AyjJMH723VRMWx5b2I9c4hri7rW1Q
import os
import io
import cv2
from google.cloud import videointelligence_v1p2beta1 as videointelligence
from difflib import SequenceMatcher
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import timeit
import collections
from PIL import Image


""" Google Function of Request Text Detection for Video from a Local File"""
"""Detect text in a local video."""


os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="VideoTextDetection-91e1475e2aea.json"



"""cut subclips
generate one clip from the video """
#
# path0 = "videos/Bullets1.mp4"
# start_time = 10*60
# end_time = start_time + 5*60
# # ffmpeg_extract_subclip(path0, start_time, end_time, targetname="testEnd10s.mp4")
# input_content = VideoFileClip(path0).subclip(start_time,end_time)

startRunning = timeit.default_timer()
"""open one  file """
path = "tempStart76-10.mp4"
# path = "videos/Bullets7.mp4"
# path = "videos/titlecardTest.png"
# path = "videos/creditCard.png"

with io.open(path, 'rb') as file:
    input_content = file.read()

print('reading: ', timeit.default_timer() - startRunning)




# video = VideoFileClip('testEnd10s.mp4')
# input_content = video.get_frame(4)
# video.save_frame('videoclipframe.png',t=4)

video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.enums.Feature.TEXT_DETECTION]
video_context = videointelligence.types.VideoContext()


operation = video_client.annotate_video(
    input_content=input_content,  # the bytes of the video file
    features=features,
    video_context=video_context)

print('\nProcessing video for text detection.')
result = operation.result(timeout=1500)

# The first result is retrieved because a single video was processed.
annotation_result = result.annotation_results[0]

# Get only the first result
text_annotation = annotation_result.text_annotations[0]



"""title similarity """
# firstresult = text_annotation.text
# print('\nText: {}'.format(text_annotation.text))
#
# similarity = SequenceMatcher(None, firstresult.lower(), "Bullets").ratio()
# print('\n similarity with Bullets {}'.format(similarity))


"""SHOW ALL RESULTS: text + timestamp """


Data = collections.namedtuple('Data', ['Text', 'Time','Confi'])

ResultList = []
# set the sort function: result will be sorted by time
def setSortedKey(p):
    return p[1]


# Show all results
for text_annotation in annotation_result.text_annotations:

    text_segment = text_annotation.segments[0]
    start_time = text_segment.segment.start_time_offset

    # if text_segment.confidence < 0.5:
    #     continue

    d = Data(text_annotation.text,start_time.seconds + start_time.nanos * 1e-9,text_segment.confidence)
    ResultList.append(d)

ResultList.sort(key = setSortedKey)
for i in ResultList:
    print('\nText: {} -  start_time: {} \nConfidence: {}'.format(i.Text.encode('utf-8'),i.Time,i.Confi))

"""--------------------------"""

#timer for running code
print('System Running Time: ', timeit.default_timer() - startRunning)

def getSimilarity(text, cast):
    cast = map(lambda x: x.lower(), cast)
    text = map(lambda x: x.lower(), text)
    right = 0;

    for tt in text:
        for s in cast:
            simi = SequenceMatcher(None, tt, s).ratio()
            if simi > 0.8:
                right += 1
                print('\n cast:{}, detect:{}, simi:{}'.format(tt, s, simi))
    return right/len(text)


# Get the first text segment
def getFirstTimeStamp(text_annotation):


    text_segment = text_annotation.segments[0]
    start_time = text_segment.segment.start_time_offset
    end_time = text_segment.segment.end_time_offset
    print('start_time: {}, end_time: {}'.format(
        start_time.seconds + start_time.nanos * 1e-9,
        end_time.seconds + end_time.nanos * 1e-9))

    print('Confidence: {}'.format(text_segment.confidence))

    # Show the result for the first frame in this segment.
    # show time offset and position
    frame = text_segment.frames[0]
    time_offset = frame.time_offset
    print('Time offset for the first frame: {}'.format(
        time_offset.seconds + time_offset.nanos * 1e-9))

    print('Rotated Bounding Box Vertices:')
    for vertex in frame.rotated_bounding_box.vertices:
        print('\tVertex.x: {}, Vertex.y: {}'.format(vertex.x, vertex.y))