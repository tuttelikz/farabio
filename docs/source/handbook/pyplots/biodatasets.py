from farabio.data.biodatasets import DSB18Dataset
train_dataset = DSB18Dataset(root="/home/data/02_SSD4TB/suzy/datasets/public/", transform=None, download=False)
dsb18_plt = train_dataset.visualize_dataset(5)
dsb18_plt.show()