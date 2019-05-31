import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="VideoTextDetection-91e1475e2aea.json"
import io
from google.cloud import videointelligence
import timeit


""" Detects camera shot changes. """
path = "videos/test20s.mp4"

startRunning = timeit.default_timer()

with io.open(path, 'rb') as file:
    input_content = file.read()

print('\nreading file time:',timeit.default_timer()- startRunning)

video_client = videointelligence.VideoIntelligenceServiceClient()
features = [videointelligence.enums.Feature.SHOT_CHANGE_DETECTION]
# operation = video_client.annotate_video(path, features=features)
operation = video_client.annotate_video(
    input_content=input_content,  # the bytes of the video file
    features=features,)
print('\nProcessing video for shot change annotations:')

result = operation.result(timeout=120)

print('\nFinished processing.')
print('\nprocess costs:',timeit.default_timer()- startRunning)

for i, shot in enumerate(result.annotation_results[0].shot_annotations):
    start_time = (shot.start_time_offset.seconds +
                  shot.start_time_offset.nanos / 1e9)
    end_time = (shot.end_time_offset.seconds +
                shot.end_time_offset.nanos / 1e9)
    print('\tShot {}: {} to {}'.format(i, start_time, end_time))



