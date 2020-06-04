## How should we annotate? 
   These annotations are only idea-forge videos.

## Which frames to pick for annotation? 

   Picking every frame for annotation is time taking and may cause baised training & testing. 
We have to pick a few images from each scene of the video. We can do this by two approaches. (few images from same background will act as natural augmentation during training,but doesn't help much during evaluation, Infact we will have biased eval benchmarks.)

1. Pick few frames from every 250 frames( 1 frame in each  10secs)
2. find separation point of scenes (Using some difference metric while comparing few conitnous frames.)


Once we have filtered  images from videos. 
1. We need annotation for bboxes around objects of concern( person for now)
2. We need flight parameters for each frame. ( optional)
   1. SLR
   2. FOV
   3. AGL
   4. TLT

We separate out videos in (80, 20)% for training and testing. 

While training in addition to default way of defining small, medium, large boxes Which are based solely on areas. We need to redefine how we keep separate buckets based on size and SLR?.  
Instead we have to separate into 
 1. small   
    1. HIGH SLR
    2. LOW SLR
 2. medium 
    1. HIGH SLR
    2. LOW SLR
 3. large
    1. HIGH SLR
    2. LOW SLR

While training model, validation mAP decides which model to pick. 

Two approaches to annotate.

In addition to annotate bbox, annotate these flight params too.

1.     After annotation of bboxes and above params
        For each bbox we can identify SLR.( Using IF code)
        We separate boxes based on SLR and FOV. 

    For each bucket of (SLR, FOV): evaluate mAP. 

2. Based only on Bbox area:

        Small ( high SLR and FOV)

        medium ( HIGH SLR and LOW FOV) (MED SLR and HIGH FOV)

        Large ( LOW SLR and all FOV)

## Observations and recommendations
Final recommendations for MVP-1
Recommendations are based on getting 0.4 mAP.
1. Analysis done only one FOV and SLR. 
   1. While FOV at 46(full zoom out)
      1. SLR < 70  Very good
      2. 70 < SLR < 120 Good
      3. 120 < SLR > 150  ( False negatives are too many)
      4. SLR > 150 not recommended
   2. While FOV at 4.6( full zoom in) 
      1. SLR < 200 Very Good.
      2. SLR < 300
      3. 300 < SLR < 350 (False negatives are many)
      4. SLR > 350 ( not recommended)




At each of these we need to find detection accuracy. 
AT FOV 46: Full zoom out 

Has 40 map at these ranges. mAp comes out to be area under PR curve. 

Max range of SLR:

AT FOV 4.6: FUll Zoom in 

Max range of SLR: 

Keep the camera aligned with the road.  towards roads/parks where people usually congregate.

SLR order at 46:

(0, 90) -> 5
(90, 120) -> 4
120, 150 -> 3
> 150 -> 2 ( High chance of False+ alert or missing alert)

(150, 200)-> 3 ( High FP)

SLR at 4.6:
Max SLR: 350 

In clear background and non-blurry images detection accuracy boosts up. 

vis_MODEL:
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.270
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.459
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.259
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.218
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.509
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.072
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.312
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.370
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.615
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.813