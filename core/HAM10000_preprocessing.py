class Ham10000Attention:

    def __init__(self, model, attention_config):
        self.config = attention_config

        self.input_shape = (self.config.resize.resizeW, self.config.resize.resizeH)
        self.model = model
        # self.attention_output = self.heatmap(self.image, model)

    def resize(self, img):
        return cv2.resize(img, self.input_shape, interpolation=cv2.INTER_LINEAR)

    def dull_razor(self, img):
        cfg = self.config.dull_razor
        if cfg.razorblur == "M":
            img = cv2.medianBlur(img, cfg.mediankernel_razorblur)
        elif cfg.razorblur == "G":
            img = cv2.GaussianBlur(img, (cfg.mediankernel_razorblur, cfg.mediankernel_razorblur), 0)

        # gyimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # filtersize = (cfg.filterstructure,cfg.filterstructure)
        # kernelrazor = cv2.getStructuringElement(cv2.MORPH_RECT, filtersize)
        # gyimage = cv2.morphologyEx(gyimage, cv2.MORPH_BLACKHAT, kernelrazor)
        #
        # retrazor, maskrazor = cv2.threshold(gyimage, cfg.lowerbound, 255, cv2.THRESH_BINARY)
        # img = cv2.inpaint(img, maskrazor, cfg.inpaintmat, cv2.INPAINT_TELEA)
        return img

    def blur(self, img):
        cfg = self.config.blur
        if cfg.normalblur == "M":
            img = cv2.medianBlur(img, cfg.mediankernel_blur)
        elif cfg.normalblur == "G":
            img = cv2.GaussianBlur(img, (cfg.mediankernel_blur, cfg.mediankernel_blur), 0)
        return img

    def softention_preprocess(self, img):
        first = preprocess_input(img)
        expanded_image = np.expand_dims(first, 0)
        return expanded_image

    def softention_mapping(self, img, LayerNumber, input_shape, SoftentionImage):
        cfg = self.config.soft_attention
        activated = self.model.predict(img)
        output = np.abs(activated)
        output = np.sum(output, axis=-1).squeeze()
        output = cv2.resize(output, input_shape)
        output /= output.max()
        # output *= 255
        # Weights =  255 - output.astype('uint8')
        #
        # heatmap = cv2.applyColorMap(Weights, cv2.COLORMAP_JET)
        # heatmap = cv2.addWeighted(heatmap, cfg.alpha, SoftentionImage, cfg.beta, cfg.gamma)
        return output

    def heatmap(self, img):
        # resized_image = self.resize(img)
        hair_removed_image = self.dull_razor(img)
        softentionImage = self.blur(hair_removed_image)
        expanded_image = self.softention_preprocess(softentionImage)
        heatmap = self.softention_mapping(expanded_image, -1, self.input_shape, softentionImage)
        return heatmap

    def preprocess(self, img):

        img = self.resize(img)
        heatmap = self.heatmap(img)
        mask = heatmap.reshape(self.config.resize.resizeW, self.config.resize.resizeH, 1)
        out = Multiply()([tf.cast(img, tf.float32), mask])
        img = tf.keras.utils.normalize(out)
        return img