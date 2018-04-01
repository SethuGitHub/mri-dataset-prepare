# mri-dataset-prepare

Part 1: Parse the DICOM images and Contour Files

How did you verify that you are parsing the contours correctly?
- By checking the parsed coordinates against the boundary of the corresponding DICOMS resolution.
- By saving the binary masks and DICOMS ans human readable images , than one to one comparison can be performed.

What changes did you make to the code, if any, in order to integrate it into our production code base?
- By Creating classes , it can be easily plugged by creating objects in the production code base.

Part 2: Model training pipeline

Did you change anything from the pipelines built in Parts 1 to better streamline the pipeline built in Part 2? If so, what? If not, is there anything that you can imagine changing in the future?
- Yes, Created Classes for both pipelines.

How do you/did you verify that the pipeline was working correctly?
- By dumping debug images and adding the debug prints.

Given the pipeline you have built, can you see any deficiencies that you would change if you had more time? If not, can you think of any improvements/enhancements to the pipeline that you could build in?
- Still need to add the code for validating the contour boundary against the dicom image boundary
- Still need to add the code for dumping debug images for masks as well as diacoms for validation
