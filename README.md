# farabio - Deep learning for biomedical imaging

Deep learning has transformed many aspects of industrial pipelines nowadays. Scientists involved in biomedical imaging research are also benefiting from the power of artificial intelligence to tackle complex challenges. Although academic community has widely accepted image analysis tools, such as scikit-image, ImageJ, there is still a need for a tool which integrates deep learning into analysis. We propose a minimal, but convenient Python package based on PyTorch with popular deep learning models, extended by trainers and datasets, providing great flexibility. Preliminary applications include classification, segmentation, detection, super-resolution, and image translation, which are crucial to perform variety of biomedical tasks.


## Installation

1. Activate conda environment:

```bash
$ conda create -n coolenv python=3.8
$ conda activate coolenv
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

2. Install prerequisites:

```bash
$ pip install -r requirements.txt
```

3. Install **farabio**:

```bash
$ pip install farabio 
```

## Documentation

You can find the API documentation on the pytorch website: https://farabio.readthedocs.io

## License

farabio has a MIT License, as found in the [LICENSE](LICENSE) file.
