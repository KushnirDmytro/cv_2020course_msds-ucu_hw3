# cv_2020course_msds-ucu_hw3
Assignment#3 for Computer Vision course 2020-2021 from Master of Data Science program, UCU (Ukraine, Lviv)

# UPD: MeanShiftTracking:
Currently dataset "RedTeam2" is tracked successfully for all 1916 frames. (with current parameters used)
Strictly requires: only first channel used, ALPHA(referrence histogram update coefficient) == 0, otherwise track will be broken due to refference histo drift.

"Walking" : solved for good extend. Requires all 3 channels (H, S of HSV channels suffer from recording aritfacts).  ALPHA>0 (0.1, 0.2 are good).

"Bike": solved almost always, but only up to the moment of tracking object moves rapidly and dissapears from a frame.

"Bolt": color-based tracking fails, as figure crosses simmilarly colored bright object, which confusess tracker.

# UPD: As dataset downloading time changes from 10min to > 1hr, decided to place final version of notebook as usual jupyter notebook. Colab version only contains changes history (if valuable) with timestamps.
Content-based image retrieval: Colab notebook: https://colab.research.google.com/drive/1ds9EVyGOoN5Xn9_5Pvmjc_lx6CqCtEd7?usp=sharing
