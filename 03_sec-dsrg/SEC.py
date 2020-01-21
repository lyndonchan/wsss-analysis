import numpy as np
import tensorflow as tf
from lib.crf import crf_inference

class SEC():
    """Class for the SEC method"""

    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset')
        self.h, self.w = (self.config.get('img_size'), self.config.get('img_size'))
        self.num_classes = self.config.get('num_classes')
        self.batch_size = self.config.get("batch_size")
        self.phase = self.config.get('phase')
        self.img_mean = self.config.get('img_mean')
        self.seed_size = self.config.get('seed_size')
        self.init_model_path = self.config.get('init_model_path', None)
        if self.dataset == 'VOC2012' or 'DeepGlobe' in self.dataset:
            self.crf_config_train = {"g_sxy":3/12,"g_compat":3,"bi_sxy":80/12,"bi_srgb":13,"bi_compat":10,"iterations":5}
            self.crf_config_test = {"g_sxy":3,"g_compat":3,"bi_sxy":80,"bi_srgb":13,"bi_compat":10,"iterations":10}
        elif self.dataset == 'ADP-morph':
            self.crf_config_train = {"g_sxy": 3 / 12, "g_compat": 3, "bi_sxy": 80 / 12, "bi_srgb": 13, "bi_compat": 10,
                                     "iterations": 5}
            self.crf_config_test = {"g_sxy": 1, "g_compat": 20, "bi_sxy": 10, "bi_srgb": 40, "bi_compat": 50,
                                     "iterations": 5}
        elif self.dataset == 'ADP-func':
            self.crf_config_train = {"g_sxy": 3 / 12, "g_compat": 3, "bi_sxy": 80 / 12, "bi_srgb": 13, "bi_compat": 10,
                                     "iterations": 5}
            self.crf_config_test = {"g_sxy": 3, "g_compat": 40, "bi_sxy": 10, "bi_srgb": 4, "bi_compat": 25,
                                    "iterations": 5}

        self.net = {}
        self.weights = {}
        self.trainable_list = []
        self.loss = {}
        self.metric = {}

        self.variables={"total":[]}

        self.min_prob = 0.0001
        self.stride = {}
        self.stride["input"] = 1

        # different lr for different variable
        self.lr_1_list = []
        self.lr_2_list = []
        self.lr_10_list = []
        self.lr_20_list = []

        self.pool = self.config.get('pool')

    def build(self,net_input=None,net_label=None,net_cues=None,net_id=None,phase='train'):
        """Build SEC model

        Parameters
        ----------
        net_input : Tensor, optional
            Input images in batch, after resizing and normalizing
        net_label : Tensor, optional
            GT segmentation in batch, after resizing
        net_cues : Tensor, optional
            Weak cue labels in batch, after resizing
        net_id : Tensor, optional
            Filenames in batch
        phase : str, optional
            Phase to run SEC model

        Returns
        -------
        (output) : Tensor
            Final layer of FCN model of SEC
        """

        if "output" not in self.net:
            if phase == 'train':
                with tf.name_scope("placeholder"):
                    self.net["input"] = net_input
                    self.net["label"] = net_label # [None, self.num_classes], int32
                    self.net["cues"] = net_cues # [None,41,41,self.num_classes])
                    self.net["id"] = net_id
                    self.net["drop_prob"] = tf.Variable(0.5, trainable=False)
            elif phase in ['val', 'tuning', 'segtest', 'test']:
                with tf.name_scope("placeholder"):
                    self.net["input"] = net_input
                    # self.net["label"] = net_label # [None, self.num_classes], int32
                    # self.net["cues"] = net_cues # [None,41,41,self.num_classes])
                    self.net["id"] = net_id
                    self.net["drop_prob"] = tf.Variable(0.0, trainable=False)
            elif phase == 'debug':
                with tf.name_scope("placeholder"):
                    self.net["input"] = net_input
                    self.net["label"] = net_label  # [None, self.num_classes], int32
                    self.net["cues"] = net_cues  # [None,41,41,self.num_classes])
                    self.net["id"] = net_id
                    self.net["drop_prob"] = tf.Variable(0.0, trainable=False)
            self.net["output"] = self.create_network(phase)
            self.pred()
        self.net["epoch"] = tf.Variable(0.0, trainable=False)
        return self.net["output"]

    def create_network(self, phase):
        """Helper function to build SEC model

        Parameters
        ----------
        phase : str, optional
            Phase to run SEC model

        Returns
        -------
        (crf) : Tensor
            Final layer of FCN model of SEC
        """
        if self.init_model_path is not None:
            self.load_init_model()
        with tf.name_scope("deeplab") as scope:
                block = self.build_block("input",["conv1_1","relu1_1","conv1_2","relu1_2","pool1"])
                block = self.build_block(block,["conv2_1","relu2_1","conv2_2","relu2_2","pool2"])
                block = self.build_block(block,["conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3"])
                block = self.build_block(block,["conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4"])
                block = self.build_block(block,["conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5","pool5a"])
                if phase == 'train':
                    fc = self.build_fc(block,["fc6","relu6","drop6","fc7","relu7","drop7","fc8"])
                elif phase in ['val', 'tuning', 'segtest', 'test']:
                    fc = self.build_fc(block, ["fc6", "relu6", "fc7", "relu7", "fc8"])
        with tf.name_scope("sec") as scope:
            softmax = self.build_sp_softmax(fc)
            crf = self.build_crf(softmax, "input")

        return self.net[crf]

    def build_block(self,last_layer,layer_lists):
        """Build a block of the SEC model

        Parameters
        ----------
        last_layer : Tensor
            The most recent layer used to build the SEC model
        layer_lists : list of str
            List of strings of layer names to build inside the current block

        Returns
        -------
        last_layer : Tensor
            The output layer of the current block
        """
        for layer in layer_lists:
            if layer.startswith("conv"):
                if layer[4] != "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
                if layer[4] == "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=2, padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
            if layer.startswith("batch_norm"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.relu( self.net[last_layer],name="relu")
                    last_layer = layer
            elif layer.startswith("pool5a"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.avg_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                    last_layer = layer
            elif layer.startswith("pool"):
                if layer[4] not in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = 2 * self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,2,2,1],padding="SAME",name="pool")
                        last_layer = layer
                if layer[4] in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                        last_layer = layer
        return last_layer

    def build_fc(self,last_layer, layer_lists):
        """Build a block of fully-connected layers

        Parameters
        ----------
        last_layer : Tensor
            The most recent layer used to build the SEC model
        layer_lists : list of str
            List of strings of layer names to build inside the current block

        Returns
        -------
        last_layer : Tensor
            The output layer of the current block
        """
        for layer in layer_lists:
            if layer.startswith("fc"):
                with tf.name_scope(layer) as scope:
                    weights,bias = self.get_weights_and_bias(layer)
                    if layer.startswith("fc6"):
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=12, padding="SAME", name="conv")

                    else:
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                    self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                    last_layer = layer
            if layer.startswith("batch_norm"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.relu( self.net[last_layer])
                    last_layer = layer
            if layer.startswith("drop"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.dropout( self.net[last_layer],rate=self.net["drop_prob"])
                    last_layer = layer

        return last_layer

    def build_sp_softmax(self,last_layer):
        """Build a block of a fully-connected layer and softmax

        Parameters
        ----------
        last_layer : Tensor
            The most recent layer used to build the SEC model

        Returns
        -------
        layer : Tensor
            The output layer of the current block
        """
        layer = "fc8-softmax"
        preds_max = tf.reduce_max(self.net[last_layer],axis=3,keepdims=True)
        preds_exp = tf.exp(self.net[last_layer] - preds_max)
        self.net[layer] = preds_exp / tf.reduce_sum(preds_exp,axis=3,keepdims=True) + self.min_prob
        self.net[layer] = self.net[layer] / tf.reduce_sum(self.net[layer],axis=3,keepdims=True)
        return layer

    def build_crf(self,featemap_layer,img_layer):
        """Build a custom dense CRF layer

        Parameters
        ----------
        featemap_layer : str
            Layer name of the feature map inputted to dense CRF layer
        img_layer : str
            Layer name of the input image

        Returns
        -------
        layer : str
            Layer name of the dense CRF layer
        """
        origin_image = self.net[img_layer] + self.img_mean
        origin_image_zoomed = tf.image.resize_bilinear(origin_image,(self.seed_size,self.seed_size))
        featemap = self.net[featemap_layer]
        def crf(featemap,image):
            batch_size = featemap.shape[0]
            image = image.astype(np.uint8)
            ret = np.zeros(featemap.shape,dtype=np.float32)
            for i in range(batch_size):
                ret[i, :, :, :] = crf_inference(image[i], self.crf_config_train, self.num_classes, featemap[i],
                                                use_log=True)
            ret[ret < self.min_prob] = self.min_prob
            ret /= np.sum(ret, axis=3, keepdims=True)
            ret = np.log(ret)
            return ret.astype(np.float32)

        layer = "crf"
        self.net[layer] = tf.py_func(crf,[featemap,origin_image_zoomed],tf.float32) # shape [N, h, w, C], RGB or BGR doesn't matter for crf
        return layer

    def load_init_model(self):
        """Load initialized layer"""
        model_path = self.config["init_model_path"]
        self.init_model = np.load(model_path,encoding="latin1").item()
		
    def get_weights_and_bias(self,layer):
        """Load saved weights and biases for saved network

        Parameters
        ----------
        layer : str
            Name of current layer

        Returns
        -------
        weights : Variable
            Saved weights
        bias : Variable
            Saved biases
        """
        if layer.startswith("conv"):
            shape = [3,3,0,0]
            if layer == "conv1_1":
                shape[2] = 3
            else:
                shape[2] = 64 * self.stride[layer]
                if shape[2] > 512: shape[2] = 512
                if layer in ["conv2_1","conv3_1","conv4_1"]: shape[2] = int(shape[2]/2)
            shape[3] = 64 * self.stride[layer]
            if shape[3] > 512: shape[3] = 512
        if layer.startswith("fc"):
            if layer == "fc6":
                shape = [3,3,512,1024]
            if layer == "fc7":
                shape = [1,1,1024,1024]
            if layer == "fc8": 
                shape = [1,1,1024,self.num_classes]
        if "init_model_path" not in self.config or self.config.get('init_model_path') is None:
            init = tf.random_normal_initializer(stddev=0.01)
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init, shape = shape)
            init = tf.constant_initializer(0)
            bias = tf.get_variable(name="%s_bias" % layer,initializer=init, shape = [shape[-1]])
        else: # restroe from init.npy
            if layer == "fc8": # using random initializer for the last layer
                init = tf.contrib.layers.xavier_initializer(uniform=True)
            else:
                init = tf.constant_initializer(self.init_model[layer]["w"])
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init,shape = shape)
            if layer == "fc8":
                init = tf.constant_initializer(0)
            else:
                init = tf.constant_initializer(self.init_model[layer]["b"])
            bias = tf.get_variable(name="%s_bias" % layer,initializer=init,shape = [shape[-1]])
        self.weights[layer] = (weights,bias)
        if layer != "fc8":
            self.lr_1_list.append(weights)
            self.lr_2_list.append(bias)
        else: # the lr is larger in the last layer
            self.lr_10_list.append(weights)
            self.lr_20_list.append(bias)
        self.trainable_list.append(weights)
        self.trainable_list.append(bias)

        return weights,bias

    def pred(self):
        """Implement final segmentation prediction as argmax of final feature map"""
        if self.h is not None:
            self.net["rescale_output"] = tf.image.resize_bilinear(self.net["output"], (self.h, self.w))
        else:
            label_size = tf.py_func(lambda x: x.shape[1:3], [self.net["input"]], [tf.int64, tf.int64])
            self.net["rescale_output"] = tf.image.resize_bilinear(self.net["output"], [tf.cast(label_size[0], tf.int32),
                                                                                       tf.cast(label_size[1],
                                                                                               tf.int32)])

        self.net["pred"] = tf.argmax(self.net["rescale_output"], axis=3)

    def getloss(self):
        """Construct overall loss function

        Returns
        -------
        loss : Tensor
            Output of overall loss function
        """
        seed_loss = self.get_seed_loss(self.net["fc8-softmax"],self.net["cues"])
        expand_loss = self.get_expand_loss(self.net["fc8-softmax"],self.net["label"])
        constrain_loss = self.get_constrain_loss(self.net["fc8-softmax"],self.net["crf"])
        self.loss["seed"] = seed_loss
        self.loss["expand"] = expand_loss
        self.loss["constrain"] = constrain_loss

        loss = seed_loss + expand_loss + constrain_loss

        return loss

    def get_seed_loss(self,softmax,cues):
        """Seeding loss function

        Parameters
        ----------
        softmax : Tensor
            Final feature map
        cues : Tensor
            Weak cues

        Returns
        -------
        loss : Tensor
            Output of seeding loss function
        """
        count = tf.math.maximum(tf.reduce_sum(cues,axis=(1,2,3),keepdims=True), 1e-5)
        loss = -tf.reduce_mean(tf.reduce_sum( cues*tf.log(softmax), axis=(1,2,3), keepdims=True)/count)
        return loss

    def get_expand_loss(self,softmax,labels):
        """Expand loss function

        Parameters
        ----------
        softmax : Tensor
            Final feature map
        labels : Tensor
            GT labels

        Returns
        -------
        loss : Tensor
            Output of expand loss function
        """
        stat = labels[:,1:]
        probs_bg = softmax[:,:,:,0]
        probs = softmax[:,:,:,1:]
        probs_max = tf.reduce_max(probs,axis=(1,2))

        q_fg = 0.996
        probs_sort = tf.contrib.framework.sort( tf.reshape(probs,(-1,self.seed_size*self.seed_size,self.num_classes-1)), axis=1)
        weights = np.array([ q_fg ** i for i in range(self.seed_size*self.seed_size -1, -1, -1)])
        weights = np.reshape(weights,(1,-1,1))
        Z_fg = np.sum(weights)
        probs_mean = tf.reduce_sum((probs_sort*weights)/Z_fg, axis=1)

        q_bg = 0.999
        probs_bg_sort = tf.contrib.framework.sort( tf.reshape(probs_bg,(-1,self.seed_size*self.seed_size)), axis=1)
        weights_bg = np.array([ q_bg ** i for i in range(self.seed_size*self.seed_size -1, -1, -1)])
        weights_bg = np.reshape(weights_bg,(1,-1))
        Z_bg = np.sum(weights_bg)
        probs_bg_mean = tf.reduce_sum((probs_bg_sort*weights_bg)/Z_bg, axis=1)

        stat_2d = tf.greater( stat, 0)
        stat_2d = tf.cast(stat_2d,tf.float32)
        self.stat = stat_2d
        
        self.loss_1 = -tf.reduce_mean( tf.reduce_sum( ( stat_2d*tf.log(probs_mean) /
                                                        tf.math.maximum(tf.reduce_sum(stat_2d,axis=1,keepdims=True), 1e-5)), axis=1))
        self.loss_2 = -tf.reduce_mean( tf.reduce_sum( ( (1-stat_2d)*tf.log(1-probs_max) /
                                                        tf.math.maximum(tf.reduce_sum((1-stat_2d),axis=1,keepdims=True), 1e-5)), axis=1))
        self.loss_3 = -tf.reduce_mean( tf.log(probs_bg_mean) )

        loss = self.loss_1 + self.loss_2 + self.loss_3
        return loss

    def get_constrain_loss(self,softmax,crf):
        """Constrain loss function

        Parameters
        ----------
        softmax : Tensor
            Final feature map
        crf : Tensor
            Output of dense CRF

        Returns
        -------
        loss : Tensor
            Output of constrain loss function
        """
        probs_smooth = tf.exp(crf)
        loss = tf.reduce_mean(tf.reduce_sum(probs_smooth * tf.log(probs_smooth/softmax), axis=3))
        return loss