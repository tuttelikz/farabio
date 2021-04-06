# farabio - Deep learning for biomedical imaging

![logo](logo/Final_Cropped_3.png)

## Abstract

Deep learning has transformed many aspects of industrial pipelines recently. Scientists involved in biomedical imaging research are also benefiting from the power of AI to tackle complex challenges. Although academic community has widely accepted image processing tools, such as scikit-image, ImageJ, there is still a need for a tool which integrates deep learning into biomedical image analysis. We propose a minimal, but convenient Python package based on PyTorch with common deep learning models, extended by flexible trainers and medical datasets.


## News
This work was selected for poster on PyTorch Ecosystem Day to be held on April 21, 2021. You can find more at: https://pytorchecosystemday.fbreg.com/

## Installation

### 1. Activate conda environment:

```bash
$ conda create -n coolenv python=3.8
$ conda activate coolenv
$ conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
```

### 2. Install prerequisites:

```bash
$ git clone https://github.com/tuttelikz/farabio.git && cd farabio
$ pip install -r requirements.txt
```

### 3. Install **farabio**:

**A. With pip**:
```bash
$ pip install farabio 
```

**B. Setup from source**:
```bash
$ pip install [-e] .      # flag for editable mode
```

## Documentation

You can find the API documentation here: https://farabio.readthedocs.io

## Contributors

Nurbolat Aimakov: [@aimakov](https://github.com/aimakov)  
Alisher Iskakov: [@finesome](https://github.com/finesome)

## License

farabio has a MIT License, as found in the [LICENSE](LICENSE) file.
