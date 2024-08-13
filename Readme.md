# Introduction
This library contains multiple small tools for Raman spectroscopy processing. Mainly targeting the mapping data collected by the Witec spectrometer. The mapping data collected by other confocal spectrometers can be converted into Witec's mapping spectral format and used with this tool.
In addition, the naming of mapping data needs to follow a fixed format to facilitate the storage of processing results.
## Naming convention for mapping files
For example:
a001,R6G,-14,1,8,8.txt

'a001': Used to record experiment numbers, can contain letters and numbers, do not include ',' and '_'. It is best not to include other special symbols, as this may cause errors.
'R6G': Name of the test sample. Please name it with letters and numbers.
'-14': The concentration. It can contain letters, numbers, '-', '.'.
'1': Integral time.
'8,8': Steps for x and y.For example, a 60 * 60 rectangular array is named '60,60'
The laser power, single step length of x and y, can be included at the position of 'A001'. The program will not process this information separately.


In the data folder, there is a txt file named 'a001, R6G, -10,1,8,8'. Please refer to this file for the mapping file format. This format can be directly exported through the official program of Witec.
# Function
## Rebaseline
The baseline removal tool is based on asymmetric least squares method.
### reBaseLine_testP_window.py
A tool with UI for attempting to remove baseline parameters. Three parameters used to estimate the baseline of asymmetric least squares method.
### reBaseLine and save_window.py
Batch processing with UI to baseline tool. You can remove the baseline of all mapping files in a folder.Then save to the specified folder.

## Draw mapping
### mapping_window.py
Draw a mapping image at the specified wavelength.
### CNN_recognition_windows.py
This program requires pre training a binary classification model for recognizing spectra using CNN_classication_train.py.
The trained CNN model with UI can recognize spectra with target features in hyperspectral data.
The following four sets of mapping images will be generated:
1. Mapping_raw specifies the mapping of intensity at a specific wavelength. The wavelength is specified in the image title.
2. Mapping_0: Probability mapping of each pixel not being the target object
3. Mapping_1: Probability mapping of each pixel as the target object
4. Mapping_p: A binary mapping that determines that the pixel of the target object is 1, not 0.

And the following three sets of data:
1. Ave_good_all is determined to be the sum of spectra and divided by the average spectrum of the total number.
2. Add the spectra judged as yes by ave_good_good and divide by the average spectrum of the total number judged as yes.
3. Take the average of all spectra in the raw file raw_rc_allmean.
### dorp_mapping_cnn.py
This program requires pre training a binary classification model for recognizing spectra using CNN_classication_train.py.

When multiple sets of mappings are collected for a concentration and placed in one folder, use this program. The program will draw each mapping in the form of subgraphs, with a default limit of 100 sets.
The following four sets of mapping images will be generated:
Mapping_raw specifies the mapping of intensity at a specific wavelength. The wavelength is specified in the image title.
Mapping_0: Probability mapping of each pixel not being the target object
Mapping_1: Probability mapping of each pixel as the target object
Mapping_p: A binary mapping that determines that the pixel of the target object is 1, not 0.

## CNN
### Manually_labeled_witec_spectra_window.py
Data tagging tool.

Select the specified mapping data file, define the result storage folder, and then manually determine whether the corresponding spectrum is the target object. Clicking on 'yes' and' no 'will save the corresponding discrimination results to the storage folder. Clicking 'Uncertain' will skip the current spectrum.
The training set created can be directly used for CNN_classification_train.py.
### CNN_classification_train.py
The training data needs to be pre labeled with Manually_labeled_witec_spectra_window.py.
Train a CNN model for Raman spectroscopy classification and recognition, binary classification. Spectral binarization can be used.
## Data conversion
### smart2witec_windows.py
Convert smart format to witec format.


