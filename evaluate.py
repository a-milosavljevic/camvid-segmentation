"""
This script is used to evaluate trained model based on the appropriate test set.
The script needs model files (e.g. trained_model.h5) to be found in "tmp" folder.
The script will produce predicted labels for test images and "evaluation.csv" file with the appropriate accuracies.
"""
from model import *
from math import ceil


# LOADING MODEL
print('Loading model')
model_path = os.path.join(tmp_folder, 'trained_model.h5')
model = tf.keras.models.load_model(model_path, compile=False)

acc_report = [['image_name', 'accuracy', 'matches', 'size']]
x_test = np.zeros((batch_size, image_height, image_width, 3), dtype=np.float32)
batch_num = ceil(len(test_images) / batch_size)
for batch_idx in range(batch_num):
    print('Batch {} / {}'.format(batch_idx + 1, batch_num))
    batch_start = batch_idx * batch_size
    batch_end = min(len(test_images), (batch_idx + 1) * batch_size)
    batch_images = test_images[batch_start:batch_end]

    # LOADING IMAGES
    print('Loading batch images')
    for i in range(len(batch_images)):
        filename_image = batch_images[i] + '.png'
        image_path = os.path.join(test_folder, filename_image)
        img = cv.imread(image_path)
        img = cv.resize(img, (image_width, image_height), interpolation=cv.INTER_LINEAR)
        x_test[i] = img

    # PREDICTING CLASSES
    print('Predicting batch classes')
    y_out = model.predict(preprocessing(x_test))
    y_out = np.argmax(y_out, axis=3).astype(np.uint8)

    # SAVING RESULTS
    print('Saving batch outputs')
    for i in range(len(batch_images)):
        filename_label = batch_images[i] + '_L.png'
        filename_prediction = batch_images[i] + '_P.png'
        label_path = os.path.join(test_labels_folder, filename_label)
        prediction_path = os.path.join(tmp_folder, filename_prediction)

        label = cv.imread(label_path)
        label_cat = bgr2cat(label)

        prediction = cv.resize(y_out[i], (label.shape[1], label.shape[0]), interpolation=cv.INTER_NEAREST)

        # Calculate metrics
        matches = np.equal(label_cat, prediction)
        acc = np.mean(matches)
        num_matches = np.sum(matches)
        num_elements = np.prod(matches.shape)
        acc_report.append([batch_images[i], acc, num_matches, num_elements])

        # Convert prediction to BGR image and save it
        prediction_bgr = cat2bgr(prediction)
        cv.imwrite(prediction_path, prediction_bgr)

# Calculate overall accuracy
acc_sum = 0
for row in acc_report[1:]:
    acc_sum += row[1]
acc_report.append(['OVERALL ACCURACY', acc_sum / (len(acc_report) - 1), '', ''])

# Save evaluation report
with open(os.path.join(tmp_folder, 'evaluation.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(acc_report)
