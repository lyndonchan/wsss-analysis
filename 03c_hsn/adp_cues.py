import warnings

from utilities import *

class ADPCues:
    """Class for handling ADP cues"""

    def __init__(self, model_name, batch_size, size, model_dir='models',
                 devkit_dir=os.path.join(os.path.dirname(os.getcwd()), 'database', 'ADPdevkit', 'ADPRelease1')):
        self.model_dir = model_dir
        self.devkit_dir = devkit_dir
        self.img_dir = os.path.join(self.devkit_dir, 'PNGImagesSubset')
        self.gt_root = os.path.join(self.devkit_dir, 'SegmentationClassAug')
        self.model_name = model_name

        self.batch_size = batch_size
        self.size = size
        self.cues = {}
        self.cues['morph'] = {}
        self.cues['func'] = {}

        # Define classes
        self.classes = {}
        if 'X1.7' in model_name:
            self.classes['all'] = ['E', 'E.M', 'E.M.S', 'E.M.U', 'E.M.O', 'E.T', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C',
                                   'C.D', 'C.D.I', 'C.D.R', 'C.L', 'H', 'H.E', 'H.K', 'H.Y', 'S', 'S.M', 'S.M.C',
                                   'S.M.S', 'S.E', 'S.C', 'S.C.H', 'S.R', 'A', 'A.W', 'A.B', 'A.M', 'M', 'M.M', 'M.K',
                                   'N', 'N.P', 'N.R', 'N.R.B', 'N.R.A', 'N.G', 'N.G.M', 'N.G.A', 'N.G.O', 'N.G.E',
                                   'N.G.R', 'N.G.W', 'N.G.T', 'G', 'G.O', 'G.N', 'T']
        else:
            self.classes['all'] = ['E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C.D.I', 'C.D.R', 'C.L',
                                   'H.E', 'H.K', 'H.Y', 'S.M.C', 'S.M.S', 'S.E', 'S.C.H', 'S.R', 'A.W', 'A.B', 'A.M',
                                   'M.M', 'M.K', 'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.W', 'G.O', 'G.N', 'T']
        self.classes['morph'] = ['E.M.S', 'E.M.U', 'E.M.O', 'E.T.S', 'E.T.U', 'E.T.O', 'E.P', 'C.D.I', 'C.D.R', 'C.L', 'H.E',
                            'H.K', 'H.Y', 'S.M.C', 'S.M.S', 'S.E', 'S.C.H', 'S.R', 'A.W', 'A.B', 'A.M', 'M.M', 'M.K',
                            'N.P', 'N.R.B', 'N.R.A', 'N.G.M', 'N.G.W']
        self.classes['valid_morph'] = ['Background'] + self.classes['morph']
        self.classes['func'] = ['G.O', 'G.N', 'T']
        self.classes['valid_func'] = ['Background', 'Other'] + self.classes['func']

        self.classinds = {}
        self.classinds['morph2valid'] = [i for i, x in enumerate(self.classes['valid_morph']) if x in self.classes['morph']]
        self.classinds['func2valid'] = [i for i, x in enumerate(self.classes['valid_func']) if x in self.classes['func']]
        self.classinds['all2morph'] = [i for i, x in enumerate(self.classes['all']) if x in self.classes['valid_morph']]
        self.classinds['all2func'] = [i for i, x in enumerate(self.classes['all']) if x in self.classes['valid_func']]
        self.classinds_arr = {}
        self.classinds_arr['morph2valid'] = np.array(self.classinds['morph2valid'])
        self.classinds_arr['func2valid'] = np.array(self.classinds['func2valid'])

        self.colours = {}
        self.colours['morph'] = get_colours('ADP-morph')
        self.colours['func'] = get_colours('ADP-func')

        self.intersects = {}
        self.intersects['morph'] = np.zeros((len(self.classes['valid_morph'])))
        self.intersects['func'] = np.zeros((len(self.classes['valid_func'])))
        self.unions = {}
        self.unions['morph'] = np.zeros((len(self.classes['valid_morph'])))
        self.unions['func'] = np.zeros((len(self.classes['valid_func'])))

    def get_img_names(self, set_name):
        """Read image names from file

        Parameters
        ----------
        set_name : str
            Name of the dataset
        """
        img_names = []
        if set_name is None:
            img_names_path = os.path.join(self.devkit_dir, 'ImageSets', 'Segmentation', 'input_list.txt')
        else:
            img_names_path = os.path.join(self.devkit_dir, 'ImageSets', 'Segmentation', set_name + '.txt')
        with open(img_names_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img_names.append(line.rstrip('\n') + '.png')
        return img_names

    def build_model(self):
        """Build CNN model from saved files"""

        # Load architecture from json
        model_json_path = os.path.join(self.model_dir, self.model_name, self.model_name + '.json')
        json_file = open(model_json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)

        # Load weights from h5
        model_h5_path = os.path.join(self.model_dir, self.model_name, self.model_name + '.h5')
        self.model.load_weights(model_h5_path)

        # Evaluate model
        opt = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

        # Load thresholds
        thresh_path = os.path.join(self.model_dir, self.model_name, self.model_name + '.mat')
        if os.path.exists(thresh_path):
            tmp = io.loadmat(thresh_path)
            self.thresholds = tmp.get('optimalScoreThresh')
        else:
            warnings.warn('No optimal thresholds found ... using 0.5 instead')
            self.thresholds = 0.5 * np.ones(self.model.output_shape[-1])

    def read_batch(self, batch_names):
        """Read batch of images from filenames

        Parameters
        ----------
        batch_names : list of str (size: B), B = batch size
            List of filenames of images in batch

        Returns
        -------
        img_batch_norm : numpy 4D array (size: B x H x W x 3), B = batch size
            Normalized batch of input images
        img_batch : numpy 4D array (size: B x H x W x 3), B = batch size
            Unnormalized batch of input images
        """
        cur_batch_size = len(batch_names)
        img_batch = np.empty((cur_batch_size, self.size, self.size, 3), dtype='uint8')
        for i in range(cur_batch_size):
            tmp = cv2.cvtColor(cv2.imread(os.path.join(self.img_dir, batch_names[i])), cv2.COLOR_BGR2RGB)
            if tmp.shape[:2] != (self.size, self.size):
                img_batch[i] = cv2.resize(tmp, (self.size, self.size))
            else:
                img_batch[i] = tmp
        img_batch_norm = (img_batch - 193.09203) / 56.450138
        return img_batch_norm, img_batch

    def get_grad_cam_weights(self, dummy_image, should_normalize=True):
        """Obtain Grad-CAM weights of the model

        Parameters
        ----------
        dummy_image : numpy 4D array (size: 1 x H x W x 3)
            A dummy image to calculate gradients
        should_normalize : bool, optional
            Whether to normalize the gradients

        Returns
        -------
        weights : numpy 2D array (size: F x C), where F = number of features, C = number of classes
            The Grad-CAM weights of the model
        """
        def find_final_layer(model):
            for iter_layer, layer in reversed(list(enumerate(model.layers))):
                if type(layer) == type(layer) == keras.layers.convolutional.Conv2D:
                    return model.layers[iter_layer + 1].name
            raise Exception('Could not find the final layer in provided HistoNet')
        self.final_layer = find_final_layer(self.model)

        conv_output = self.model.get_layer(self.final_layer).output  # activation_7
        num_classes = self.model.output_shape[1]
        num_feats = int(conv_output.shape[-1])
        weights = np.zeros((num_feats, num_classes))

        def normalize(x):
            # utility function to normalize a tensor by its L2 norm
            return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

        for iter_class in range(self.model.output_shape[1]):
            y_c = self.model.layers[-2].output[0, iter_class]
            if should_normalize:
                grad = normalize(K.gradients(y_c, conv_output)[0])
            else:
                grad = K.gradients(y_c, conv_output)[0]
            grad_func = K.function([self.model.layers[0].input, K.learning_phase()], [conv_output, grad])
            conv_val, grad_val = grad_func([dummy_image, 0])
            conv_val, grad_val = conv_val[0], grad_val[0]
            weights[:, iter_class] = np.mean(grad_val, axis=(0, 1))
        return weights