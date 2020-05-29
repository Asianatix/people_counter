# People detector model

## Install 

## To Run 
Need one or two checkpoints based on whether deepsort is turned on/off      
People detector model:      
deepsort checkpoint:        

### Real time 
This is more focussed towards real rtsp stream and real time predictions.
```
    python pc.py  
        --video_path <path to video>  
        --real_time  
        ``` For different processing policy
            --capture_buffer_length <5>   
            --buffer_filter_policy  <policy_name>
            --apply_batch_ensemble  
        ```
        ``` different methods for saving predictions 
            --tcp_ip_port <127.0.0.1:9999>
            --save_video_to <path to folder>
            --save_frames_to <path to folder>
        ```
```

### On normal videos
This is for offline predictions
```
    python pc.py   
        --video_path <path to video>  
        --deep_sort # Use this when running on continous frames
        ``` different methods for saving predictions 
            --tcp_ip_port <127.0.0.1:9999>
            --save_video_to <path to folder>
            --save_frames_to <path to folder>
        ```
```

### generate annotations 

Use this to generate model predictions and save them in Labelme format
```
    python to_labelme.py
```    



## TODO 

[] Run on images
[] 