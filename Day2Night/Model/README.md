# Model
  This part focuses on the training part of the model. The theory part can be found [here](/Base_Model/README.md).

  The [dataset](Dataset/) is pre-processed - random cropping, jittering and normalization.

  The `BUFFER_SIZE` is set to maximum, i.e., the no. of training images. I have trained the model with less than 2000 buffer size, as
  to reduce RAM usage and GPU load.

  The model, then, can be trained by running the script. Every 5000 steps, the model is saved. The training can be continued
  by uncommenting `checkpoint.restore(manager.latest_checkpoint)`.
