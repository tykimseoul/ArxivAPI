from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.backend import *
from tensorflow.keras.regularizers import l2


def tversky(y_true, y_pred, smooth=1, alpha=0.7):
    y_true_pos = flatten(y_true)
    y_pred_pos = flatten(y_pred)
    true_pos = sum(y_true_pos * y_pred_pos)
    false_neg = sum(y_true_pos * (1 - y_pred_pos))
    false_pos = sum((1 - y_true_pos) * y_pred_pos)
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky_loss(y_true, y_pred, gamma=0.75):
    tv = tversky(y_true, y_pred)
    return pow((1 - tv), gamma)


class Unet:
    def __init__(self, num_class, input_size=(512, 512, 1), batch_size=1, deep_supervision=False):
        self.base_model = None
        self.input_size = input_size
        self.num_class = num_class
        self.batch_size = batch_size
        self.deep_supervision = deep_supervision
        self.model = self.build()

    def build(self):
        def conv_block(inputs, filters, kernel_size, batch_normalization=True, residual=True):
            conv = Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
            if batch_normalization:
                conv = BatchNormalization()(conv)
            conv = Conv2D(filters, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
            if batch_normalization:
                conv = BatchNormalization()(conv)
            if residual:
                shortcut = Conv2D(filters, 1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
                shortcut = BatchNormalization()(shortcut)
                conv = Add()([shortcut, conv])
            return conv

        inputs = Input(self.input_size)

        conv0_0 = conv_block(inputs, 32, 3)
        pool0 = MaxPooling2D(pool_size=(2, 2))(conv0_0)

        conv1_0 = conv_block(pool0, 64, 3)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_0)

        conv2_0 = conv_block(pool1, 128, 3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_0)

        conv3_0 = conv_block(pool2, 256, 3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_0)

        conv4_0 = conv_block(pool3, 512, 3)

        up0_1 = Conv2DTranspose(32, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv1_0)
        conv0_1 = Concatenate()([up0_1, conv0_0])
        conv0_1 = conv_block(conv0_1, 32, 3)

        up1_1 = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2_0)
        conv1_1 = Concatenate()([up1_1, conv1_0])
        conv1_1 = conv_block(conv1_1, 64, 3)

        up2_1 = Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv3_0)
        conv2_1 = Concatenate()([up2_1, conv2_0])
        conv2_1 = conv_block(conv2_1, 128, 3)

        up3_1 = Conv2DTranspose(256, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv4_0)
        conv3_1 = Concatenate()([up3_1, conv3_0])
        conv3_1 = conv_block(conv3_1, 256, 3)

        up0_2 = Conv2DTranspose(32, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv1_1)
        conv0_2 = Concatenate()([up0_2, conv0_0, conv0_1])
        conv0_2 = conv_block(conv0_2, 32, 3)

        up1_2 = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2_1)
        conv1_2 = Concatenate()([up1_2, conv1_0, conv1_1])
        conv1_2 = conv_block(conv1_2, 64, 3)

        up2_2 = Conv2DTranspose(128, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv3_1)
        conv2_2 = Concatenate()([up2_2, conv2_0, conv2_1])
        conv2_2 = conv_block(conv2_2, 128, 3)

        up0_3 = Conv2DTranspose(32, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv1_2)
        conv0_3 = Concatenate()([up0_3, conv0_0, conv0_1, conv0_2])
        conv0_3 = conv_block(conv0_3, 32, 3)

        up1_3 = Conv2DTranspose(64, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2_2)
        conv1_3 = Concatenate()([up1_3, conv1_0, conv1_1, conv1_2])
        conv1_3 = conv_block(conv1_3, 64, 3)

        up0_4 = Conv2DTranspose(32, 2, strides=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv1_3)
        conv0_4 = Concatenate()([up0_4, conv0_0, conv0_1, conv0_2, conv0_3])
        conv0_4 = conv_block(conv0_4, 32, 3)

        nestnet_output_1 = Conv2D(self.num_class, (1, 1), activation='sigmoid', name='output_1', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv0_1)
        nestnet_output_2 = Conv2D(self.num_class, (1, 1), activation='sigmoid', name='output_2', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv0_2)
        nestnet_output_3 = Conv2D(self.num_class, (1, 1), activation='sigmoid', name='output_3', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv0_3)
        nestnet_output_4 = Conv2D(self.num_class, (1, 1), activation='sigmoid', name='output_4', kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(conv0_4)

        if self.deep_supervision:
            output = [nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4]
        else:
            output = nestnet_output_4

        model = Model(inputs, output)

        model.compile(optimizer=Adam(lr=1e-4), loss=focal_tversky_loss, metrics=[tversky])

        model.summary()

        return model
