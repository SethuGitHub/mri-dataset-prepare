import pydicom
from pydicom.errors import InvalidDicomError
import numpy as np
from PIL import Image, ImageDraw
import pandas as pd
import os
import glob
import errno
import shutil
import png
from numpy import random
import json


class prepare_dataset_pipeline :

    def __init__(self, json_params):
        """Init the prepare dataset pipeline
        :param json_params: json parameters with all the input and output data paths
        :return: None
        """
        params = json.loads(json_params)

        self.link_file_path = params['link_file_path']
        self.dicoms_path = params['dicoms_path']
        self.contourfiles_path = params['contourfiles_path']
        self.contourfiles_type = params['contourfiles_type']
        self.boolean_mask_dir = params['boolean_mask_dir']
        self.valid_dicoms_dir = params['valid_dicoms_dir']

    def load_matching_dicom_contour_files(self):
        """Read CSV file to get the list of DICOMS and contour file's
        :param self: self object
        :return: dicom,contour : list DICOM and Contour files ater one to one mapping
        """
        #
        link_file_loc = self.link_file_path
        dicoms_loc = self.dicoms_path
        contourfiles_loc = self.contourfiles_path
        contourfiles_type = self.contourfiles_type

        dicom = []
        contour = []

        link = pd.read_csv(link_file_loc)

        print "\n*** Count of given DICOM and Contour files ***"
        for index, row in link.iterrows():
            # Form the folder paths for dicoms and contour files
            dicoms_path = os.path.join(dicoms_loc,str((row['patient_id'])))
            contourfiles_path = os.path.join(contourfiles_loc,str((row['original_id'])),contourfiles_type)
            dicom_files = glob.glob(dicoms_path + "/*.dcm")  # list of dicom files
            contour_files = glob.glob(contourfiles_path + "/*.txt")  # list of contour files
            print dicoms_path,len(dicom_files) ,contourfiles_path ,len(contour_files)
            # Check for one to one mapping
            # Iterate each contour file check for the corresponding dicom file
            for contour_file in contour_files:
                dcm_name = str(int(os.path.basename(contour_file).split("-")[2])) + ".dcm"
                dcm_file_path = os.path.join(dicoms_path,dcm_name)
                if os.path.exists(dcm_file_path) :
                    dicom.append(dcm_file_path)
                    contour.append(contour_file)
                else:
                    print "File Not Exist : " + os.path.join(dicoms_path,dcm_name)

        print "\n*** Count after one to one mapping for DICOM and Contour files ***"
        print "Valid Dicoms :- ",len(dicom) , "Given Contour Files :- " ,len(contour)
        return dicom,contour


    def parse_contour_file(self,filename):
        """Parse the given contour filename
        :param filename: filepath to the contourfile to parse
        :return: list of tuples holding x, y coordinates of the contour
        """
        coords_lst = []
        if os.path.exists(filename):
            with open(filename, 'r') as infile:
                for line in infile:
                    coords = line.strip().split()
                    x_coord = float(coords[0])
                    y_coord = float(coords[1])
                    coords_lst.append((x_coord, y_coord))
        else:
            print "File Not Exist : " + filename
        return coords_lst


    def poly_to_mask(self,contour_file,polygon, width, height):
        """Convert polygon to mask
        :param polygon: list of pairs of x, y coords [(x1, y1), (x2, y2), ...]
         in units of pixels
        :param width: scalar image width
        :param height: scalar image height
        :return: Boolean mask of shape (height, width)
        """
        # http://stackoverflow.com/a/3732128/1410871
        img = Image.new(mode='L', size=(width, height), color=0)
        ImageDraw.Draw(img).polygon(xy=polygon, outline=0, fill=1)
        #img.save(str(int(os.path.basename(contour_file).split("-")[2])) + ".jpg", 'JPEG') # For debugging
        mask = np.array(img).astype(bool)
        return mask


    def create_boolean_masks(self,contour_file_lst, width, height):
        """ Create the boolean masks for the selected  contour file's
        :param contour_file_lst, width, height: selected contour file list , width and height of diacoms
        :return: boolean_mask_lst : list of boolean masks created
        """
        try:
            if os.path.exists(self.boolean_mask_dir):
                shutil.rmtree(self.boolean_mask_dir)
            os.makedirs(self.boolean_mask_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        boolean_mask_lst = []
        for file in contour_file_lst:
            try:
                npfile_folder = os.path.split(file)[0].split("/")[1]
                npfile_folder_sub_dir = os.path.join(self.boolean_mask_dir,npfile_folder)
                os.makedirs(npfile_folder_sub_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            mask_points_lst = self.parse_contour_file(file)
            bmask = self.poly_to_mask(file, mask_points_lst, width, height)
            npfile = os.path.basename(file).split("-")[2]
            outfile = os.path.join(npfile_folder_sub_dir, '%s.npy' % npfile)
            np.save(outfile, bmask)
            boolean_mask_lst.append(outfile)
        return boolean_mask_lst


    def parse_dicom_file(self,filename):
        """Parse the given DICOM filename
        :param filename: filepath to the DICOM file to parse
        :return: image_2d_scale,dcm_width,dcm_height : scaled pixel array of diacom file, width and height of diacoms
        """
        try:
            dcm = pydicom.read_file(filename)
            dcm_image = dcm.pixel_array
            dcm_width = dcm.Columns
            dcm_height = dcm.Rows
            try:
                intercept = dcm.RescaleIntercept
            except AttributeError:
                intercept = 0.0
            try:
                slope = dcm.RescaleSlope
            except AttributeError:
                slope = 0.0

            if intercept != 0.0 and slope != 0.0:
                dcm_image = dcm_image*slope + intercept
            #dcm_dict = {'pixel_data' : dcm_image}

            # Convert to float to avoid overflow or underflow losses.
            image_2d = dcm.pixel_array.astype(float)

            # Rescaling grey scale between 0-255
            image_2d_scale = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0

            # Convert to uint
            image_2d_scale = np.uint8(image_2d_scale)
            return image_2d_scale,dcm_width,dcm_height

        except InvalidDicomError:
            return None

    def create_valid_dicoms(self,contour_file_lst,dicoms_lst):
        """ Check for the valid diacoms
        :param contour_file_lst,dicoms_lst: list of contour files , list of diacom  files
        :return: valid_dicoms_lst,dcm_width, dcm_height : return the valid diacoms , width and height of diacoms
        """
        prev_width = -1
        prev_height = -1

        try:
            if os.path.exists(self.valid_dicoms_dir):
                shutil.rmtree(self.valid_dicoms_dir)
            os.makedirs(self.valid_dicoms_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        valid_dicoms_lst= []

        for file in dicoms_lst:
            try:
                npfile_folder = os.path.split(file)[0].split("/")[1]
                # print npfile_folder
                npfile_folder_sub_dir = os.path.join(self.valid_dicoms_dir, npfile_folder)
                os.makedirs(npfile_folder_sub_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
            try :
                # Parse each dicom image
                image_2d_scaled, dcm_width, dcm_height = self.parse_dicom_file(file)

                # Detect if any difference in dicom image width and height
                if (prev_width ==-1) and (prev_height == -1):
                    prev_width, prev_height = dcm_width, dcm_height
                    #print dcm_width, dcm_height
                elif (not(prev_width == dcm_width)) or (not(prev_width == dcm_width)):
                    print "Dicom Width and Height is not same as previous!!"
                    print dcm_width, dcm_height
            except InvalidDicomError:
                return None

            prev_width , prev_height = dcm_width, dcm_height

            # Save dicom pixel array as np array
            npfile = os.path.basename(file).split("/")[0].split(".")[0]
            outfile = os.path.join(npfile_folder_sub_dir, '%s.npy' % npfile)
            np.save(outfile, image_2d_scaled)


            # Writing the PNG file to verify the dicoms
            w = png.Writer(dcm_width,dcm_height, greyscale=True)
            png_file_path = os.path.join(npfile_folder_sub_dir, '%s.png' % npfile)
            png_file = open(png_file_path, 'wb')
            w.write(png_file, image_2d_scaled)
            png_file.close()


            valid_dicoms_lst.append(outfile)

        return valid_dicoms_lst,dcm_width, dcm_height


    # def validate_contour_files(dicoms_loc,contourfiles_loc,contourfiles_type):
    #       Check the resolution of dicom  image
    #       Check the validity of the contour boundary against the dicom boundary
    #       Eliminate the odds and save the results in the separate folder


class train_pipeline:

    def __init__(self, json_params):
        """Init the prepare train pipeline
        :param json_params: json parameters with all the train params
        :return: None
        """
        params = json.loads(json_params)
        self.batch_size = int(params['batch_size'])
        self.nb_epoch = int(params['nb_epoch'])

    def get_random_mini_batches(self,data,target):
        """ Create the random mini batches from entire dataset
        :param data,target: input data and target numpy array
        :return: mini_batches :randomly created mini batch
        """
        random_idxs = random.choice(len(target), len(target), replace=False)
        data_shuffled = data[random_idxs,:]
        target_shuffled = target[random_idxs]
        mini_batches = [(data_shuffled[i:i+self.batch_size,:], target_shuffled[i:i+self.batch_size]) for i in range(0, len(target),self.batch_size)]
        return mini_batches

    def train_model(self,valid_dicom_npy_lst, boolean_mask_lst):
        """ load batches of data for input into a 2D deep learning model
        :param valid_dicom_npy_lst, boolean_mask_lst: saved information from the DICOM images and contour files masks
        :return: None
        """
        train_data = []
        train_target = []

        # Using the  saved numpy information files from the DICOM images and contour files,
        for dicom_npy_file in valid_dicom_npy_lst:
           train_data.append(np.load(dicom_npy_file))

        for mask_npy_file in boolean_mask_lst:
           train_target.append(np.load(mask_npy_file))

        # Forming the single numpy array for the entire data set
        train_data = np.array(train_data, dtype=np.float32)
        train_target = np.array(train_target, dtype=np.float32)

        for i in range(self.nb_epoch) :
            print('Epoch {} of {}'.format(i, self.nb_epoch))
            # get the single batch of train data consists of one numpy array for images & one numpy array for targets.
            mini_random_batches = self.get_random_mini_batches(train_data, train_target)
            print 'Random Mini Batch Length : {} out of {}  entire samples'.format(mini_random_batches.__len__(),train_data.__len__())
            for random_batch in mini_random_batches:
                train_data_batch = random_batch[0]
                train_target_batch = random_batch[1]



if __name__ == "__main__":

    dataset_prep_params = {
        'link_file_path': 'link.csv',         # Specify the full path of link file
        'dicoms_path': 'dicoms',              # Specify the full path of dicoms
        'contourfiles_path': 'contourfiles',  # Specify the full path of contour files
        'contourfiles_type': 'i-contours',    # Select  the  'i-contours' or  'o-contours'
        'boolean_mask_dir': 'boolean_masks',  # Specify the full path of boolean masks to be saved
        'valid_dicoms_dir': 'valid_dicoms',   # Specify the full path of valid dicoms to be saved
       }

    # PART-1  : Pipe line to Parse the DICOM images and Contour Files and save valid information for train pipeline

    jsonData = json.dumps(dataset_prep_params)

    # Create the prepare data set pipeline object
    prep_data_pipe = prepare_dataset_pipeline(jsonData)

    # Load Dicoms and Contour files with one to one mapping
    dicoms_lst, contour_file_lst =  prep_data_pipe.load_matching_dicom_contour_files()

    # Create Valid diacoms list
    valid_dicom_npy_lst,width ,height = prep_data_pipe.create_valid_dicoms(contour_file_lst,dicoms_lst)

    # Create boolean masks list
    boolean_mask_lst = prep_data_pipe.create_boolean_masks(contour_file_lst,width,height)

    # Validate the correctness of selected contour files
    # def validate_contour_files(dicoms_loc,contourfiles_loc,contourfiles_type)



    # PART-2  : Model training pipeline

    train_params = {
        'batch_size': '8',  # Specify the mini random batch size from cycle through entire data set
        'nb_epoch': '100'   # Specify the number of epochs
       }

    jsonData = json.dumps(train_params)

    # Create the training pipeline  object
    train_pipe = train_pipeline(jsonData)

    # Load batches of data for input into a 2D deep learning model
    train_pipe.train_model(valid_dicom_npy_lst, boolean_mask_lst)
