import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import time

class TrainPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, img_root, shuffle = True):
        super(TrainPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_root = img_root, random_shuffle=shuffle)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.res = ops.RandomResizedCrop(device="cpu", size=224, random_area=[0.2, 1.0])
        self.cmnp = ops.CropMirrorNormalize(device="cpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            mean=[0.485*255, 0.456*255, 0.406*255],
                                            std=[0.229*255, 0.224*255, 0.225*255])
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input(name="TrainReader")
        images = self.decode(jpegs)
        images = self.res(images)
        images = self.cmnp(images, mirror=rng)
        return (images, labels)

class TestPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, img_root, shuffle = False):
        super(TestPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(file_root = img_root, random_shuffle=shuffle)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.res = ops.Resize(device="cpu", resize_shorter=256)
        self.cmnp = ops.CropMirrorNormalize(device="cpu",
                                            dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            mirror=0,
                                            mean=[0.485*255, 0.456*255, 0.406*255],
                                            std=[0.229*255, 0.224*255, 0.225*255])

    def define_graph(self):
        jpegs, labels = self.input(name="TestReader")
        images = self.decode(jpegs)
        images = self.res(images)
        images = self.cmnp(images)
        return (images, labels)


if __name__ == '__main__':
    image_dir = "../../dataset/imagenet/train"
    label_range = (0, 999)
    pipe = TrainPipeline(batch_size=256, num_threads=6, device_id = 0, img_root = image_dir)
    pipe.build()
    dali_iter = DALIGenericIterator(pipe, ['data', 'label'], pipe.epoch_size("TrainReader"))
    start = time.time()
    for i, data in enumerate(dali_iter):
        print('idx: ',i)
        # Testing correctness of labels
        images = data[0]["data"].cuda(non_blocking=True)
        for d in data:
            label = d["label"]
            print('label: ',label)
            image = d["data"]
            ## labels need to be integers
            assert(np.equal(np.mod(label, 1), 0).all())
            ## labels need to be in range pipe_name[2]
            assert((label >= label_range[0]).all())
            assert((label <= label_range[1]).all())
    print("OK, time:",time.time() - start)
