import mediapipe         as mp

class PoseDetector:

    def __init__(self):
    
        self.pose = mp.solutions.pose.Pose(
            static_image_mode        = True,
            model_complexity         = 2,
            enable_segmentation      = True,
            min_detection_confidence = 0.5,
        )

        return


    def getResults( self, image ):
        results = self.pose.process( image )
        return results

    
