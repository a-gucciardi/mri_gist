import utils as ut
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX') 

ds_folds = ut.siamese_noise_dataset_fold2(test_size=0.2, noise_size='lil', n_splits=5)
im = ds_folds[0]['train']['slice1'][0]

print(type)

# im.show()

plt.imshow(im)
plt.axis('off')
plt.show()