from keras import optimizers
from keras.models import model_from_json
import scipy.io as io
import warnings

from utilities import *

class ADPCues:
    """Class for handling ADP cues"""

    def __init__(self, model_name, batch_size, size, model_dir='models',
                 devkit_dir=os.path.join(os.path.dirname(os.getcwd()), 'database', 'ADPdevkit', 'ADPRelease1')):
        self.model_dir = model_dir
        self.devkit_dir = devkit_dir
        self.img_dir = os.path.join(self.devkit_dir, 'JPEGImages')
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
        self.predicted_totals = {}
        self.predicted_totals['morph'] = np.zeros((len(self.classes['valid_morph'])))
        self.predicted_totals['func'] = np.zeros((len(self.classes['valid_func'])))
        self.gt_totals = {}
        self.gt_totals['morph'] = np.zeros((len(self.classes['valid_morph'])))
        self.gt_totals['func'] = np.zeros((len(self.classes['valid_func'])))

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
            with open(img_names_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip("\n")
                    name, _ = line.split(' ')
                    img_names.append(name)
        else:
            img_names_path = os.path.join(self.devkit_dir, 'ImageSets', 'Segmentation', set_name + '.txt')
            with open(img_names_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.rstrip("\n") + '.jpg'
                    img_names.append(line)
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

    def grad_cam(self, weights, images, is_pass_threshold, orig_sz=[224, 224],
                 should_upsample=False):
        """Generate Grad-CAM

        Parameters
        ----------
        weights : numpy 2D array (size: F x C), where F = number of features, C = number of classes
            The Grad-CAM weights of the model
        images : numpy 4D array (size: B x H x W x 3), where B = batch size
            The batch of input images
        is_pass_threshold : numpy 2D bool array (size: B x C), where B = batch size, C = number of classes
            An array saving which classes pass the pre-defined thresholds for each image in the batch
        orig_sz : list of int, optional
            2D size of original images

        Returns
        -------
        cams_thresh : numpy 4D array (size: B x H x W x C), B = batch size, C = number of classes
            The thresholded Grad-CAMs
        """
        conv_output = self.model.get_layer(self.final_layer).output  # activation_7
        conv_func = K.function([self.model.layers[0].input], [conv_output])
        conv_val = conv_func([images])
        conv_val = conv_val[0]
        cams = np.maximum(np.einsum('ijkl,lm->ijkm', conv_val, weights), 0)
        if should_upsample:
            old_cams = cams[:]
            cams = np.zeros((old_cams.shape[0], orig_sz[0], orig_sz[1], old_cams.shape[-1]))
            for i in range(cams.shape[0]):
                for j in range(cams.shape[-1]):
                    cams[i, :, :, j] = cv2.resize(cams[i, :, :, j], (orig_sz[0], orig_sz[1]))
        cams_thresh = cams * np.expand_dims(np.expand_dims(is_pass_threshold, axis=1), axis=2)
        return cams_thresh

    def split_by_httclass(self, H):
        """Split classes in incoming variable by HTT class

        Parameters
        ----------
        H : numpy <=2D array (size: B x C x ?), where B = batch size, C = number of classes
            Variable to be split

        Returns
        -------
        (H_morph) : numpy <=2D array (size: B x C_morph x ?), where B = batch size, C_morph = number of morphological classes
            Split morphological classes in variable
        (H_func) : numpy <=2D array (size: B x C_func x ?), where B = batch size, C_morph = number of functional classes
            Split functional classes in variable
        """
        morph_all_inds = [i for i, x in enumerate(self.classes['all']) if x in self.classes['morph']]
        func_all_inds = [i for i, x in enumerate(self.classes['all']) if x in self.classes['func']]
        return H[:, morph_all_inds], H[:, func_all_inds]

    def modify_by_htt(self, gradcam, images, classes, gradcam_adipose=None):
        """Generates non-foreground class activations and appends to the foreground class activations

        Parameters
        ----------
        gradcam : numpy 4D array (size: self.batch_size x C x H x W), where C = number of classes
            The serialized Grad-CAM for the current batch
        images : numpy 3D array (size: self.batch_size x H x W x 3)
            The input images for the current batch
        classes : list (size: C), where C = number of classes
            The list of classes in gradcam
        gradcam_adipose : numpy 4D array (size: self.num_imgs x C x H x W), where C = number of classes,
                          or None, optional
            Adipose class Grad-CAM (if segmenting functional types) or None (if not segmenting functional types)

        Returns
        -------
        gradcam : numpy 4D array (size: self.batch_size x C x H x W), where C = number of classes
            The modified Grad-CAM for the current batch, with non-foreground class activations appended
        """
        if gradcam_adipose is None:
            htt_class = 'morph'
        else:
            htt_class = 'func'
        if htt_class == 'morph':
            background_max = 0.75
            background_exception_classes = ['A.W', 'A.B', 'A.M']
        elif htt_class == 'func':
            background_max = 0.75
            other_tissue_mult = 0.05
            background_exception_classes = ['G.O', 'G.N', 'T']
            if gradcam_adipose is None:
                raise Exception('You must feed in adipose heatmap for functional type')
            other_ind = classes.index('Other')
        background_ind = classes.index('Background')

        # Get background class prediction
        mean_img = np.mean(images, axis=-1)
        sigmoid_input = 4 * (mean_img - 240)
        background_gradcam = background_max * scipy.special.expit(sigmoid_input)
        background_exception_cur_inds = [i for i, x in enumerate(classes) if x in background_exception_classes]
        for iter_input_image in range(background_gradcam.shape[0]):
            background_gradcam[iter_input_image] = scipy.ndimage.gaussian_filter(background_gradcam[iter_input_image],
                                                                                 sigma=2)
        if background_gradcam.shape[1] != gradcam.shape[2] or background_gradcam.shape[2] != gradcam.shape[3]:
            old_bg = background_gradcam[:]
            background_gradcam = np.zeros((old_bg.shape[0], gradcam.shape[2], gradcam.shape[3]))
            for i in range(background_gradcam.shape[0]):
                background_gradcam[i] = cv2.resize(old_bg[i], (gradcam.shape[2], gradcam.shape[3]))
        background_gradcam -= np.max(gradcam[:, background_exception_cur_inds], axis=1)
        gradcam[:, background_ind] = background_gradcam

        # Get other tissue class prediction
        if htt_class == 'func':
            other_moh = np.max(gradcam, axis=1)
            other_gradcam = np.expand_dims(other_tissue_mult * (1 - other_moh), axis=1)
            other_gradcam = np.max(np.concatenate((other_gradcam, gradcam_adipose), axis=1), axis=1)
            gradcam[:, other_ind] = other_gradcam
        return gradcam

    def update_cues(self, gradcam, class_inds, htt_class, indices, thresh):
        """Update the cues class object with current batch's Grad-CAM

        Parameters
        ----------
        gradcam : numpy 4D array (size: self.batch_size x C x H x W), where C = number of classes
            The serialized Grad-CAM for the current batch
        class_inds : numpy 1D array (size: self.batch_size)
            List of image indices in batch, as array
        htt_class : str
            The type of segmentation set to solve
        indices : list of int (size: self.batch_size)
            List of image indices in batch
        thresh: float
            Confidence value for thresholding activation maps [0-1]
        """
        localization_onehot = np.zeros_like(gradcam)
        # Non-other
        localization = np.array(gradcam > thresh *
                                np.expand_dims(np.expand_dims(np.max(gradcam, axis=(2, 3)), axis=2), axis=3))

        # Solve overlap conflicts
        class_rank = np.argsort(-np.sum(np.sum(localization, axis=-1), axis=-1))  # from largest to smallest masks
        localization_ind = np.zeros((gradcam.shape[0], gradcam.shape[2], gradcam.shape[3]), dtype='int64')
        img_inds = np.arange(class_rank.shape[0])
        for iter_class in range(class_rank.shape[1]):
            cur_masks = localization[img_inds, class_rank[:, iter_class]]
            localization_ind *= ~cur_masks
            localization_ind += np.expand_dims(np.expand_dims(class_rank[:, iter_class] + 1, axis=1),
                                               axis=2) * cur_masks
        for iter_class in range(class_rank.shape[1]):
            localization_onehot[:, iter_class] = localization_ind == (iter_class + 1)
        # Save true one-hot encoded values
        for i, x in enumerate(indices):
            self.cues[htt_class]['%d_labels' % x] = class_inds[i]
            self.cues[htt_class]['%d_cues' % x] = np.array(np.where(localization_onehot[i]))  # class is front

    def read_gt_batch(self, htt_class, batch_names):
        """Read batch of GT segmentation images

        Parameters
        ----------
        htt_class : str
            The type of segmentation set to solve
        batch_names : list of str (size: B), B = batch size
            List of filenames of images in batch

        Returns
        -------
        gt_batch : numpy 4D array (size: B x H x W x 3), where B = batch size
            The batch of GT segmentation images
        """
        cur_batch_size = len(batch_names)
        gt_batch = np.empty((cur_batch_size, self.size, self.size, 3), dtype='uint8')
        batch_names = [os.path.splitext(x)[0] + '.png' for x in batch_names]
        for i in range(cur_batch_size):
            tmp = cv2.cvtColor(cv2.imread(os.path.join(self.gt_root, 'ADP-' + htt_class, batch_names[i])),
                               cv2.COLOR_BGR2RGB)
            if tmp.shape[:2] != (self.size, self.size):
                gt_batch[i] = cv2.resize(tmp, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return gt_batch