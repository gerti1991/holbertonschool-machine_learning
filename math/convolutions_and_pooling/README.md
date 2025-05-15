# Convolutions and Pooling

## What is a Convolution?
A **convolution** is a mathematical operation used in image processing and deep learning to extract features from data. In the context of images, it involves sliding a small matrix (called a kernel or filter) over the image and computing a weighted sum of the overlapping values. This operation helps detect patterns such as edges, textures, and shapes.

---

## What is a Kernel/Filter?
A **kernel** (or **filter**) is a small matrix of weights, typically of size 3x3, 5x5, etc. It is used to transform the input image by emphasizing certain features. The kernel is applied to each region of the image, and the result is a new, filtered image.

---

## What is Padding?
**Padding** refers to adding extra pixels (usually zeros) around the border of an image before applying a convolution. Padding helps control the spatial size of the output and preserves information at the edges of the image.

- **No padding**: The output shrinks after convolution.
- **Zero padding**: The output size can be preserved or controlled.

### "Same" Padding
- Output size is the same as the input size.
- Padding is added so the kernel fits all positions.

### "Valid" Padding
- No padding is added.
- The kernel only slides where it fully overlaps with the image.
- Output is smaller than the input.

---

## What is a Stride?
**Stride** is the number of pixels the kernel moves each time it slides over the image. A stride of 1 means the kernel moves one pixel at a time; a stride of 2 means it moves two pixels, and so on. Larger strides reduce the output size and computation.

---

## What are Channels?
**Channels** refer to the number of color components in an image:
- **Grayscale** images have 1 channel.
- **RGB** images have 3 channels (Red, Green, Blue).
- Each channel is convolved separately, and the results are combined.

---

## How to Perform a Convolution Over an Image
1. Place the kernel at the top-left corner of the image.
2. Multiply each kernel value by the corresponding image pixel.
3. Sum all the multiplications to get a single output value.
4. Move the kernel according to the stride and repeat.
5. Continue until the kernel has covered the entire image.

**Example:**
```
Input Image (5x5):
1 2 3 4 5
6 7 8 9 0
1 3 5 7 9
2 4 6 8 0
1 1 1 1 1

Kernel (3x3):
1 0 1
0 1 0
1 0 1
```

---

## What is Max Pooling? Average Pooling?
**Pooling** is a downsampling operation that reduces the spatial size of the feature maps and helps make the representation more robust.

- **Max Pooling**: Takes the maximum value from each patch of the feature map.
- **Average Pooling**: Takes the average value from each patch.

Both are typically applied with a window (e.g., 2x2) and a stride.

---

## How to Perform Max/Average Pooling Over an Image
1. Define a pooling window size (e.g., 2x2) and stride.
2. Slide the window over the input feature map.
3. For each window:
   - **Max pooling**: Output the maximum value.
   - **Average pooling**: Output the average value.
4. Continue until the window has covered the entire feature map.

**Example:**
```
Input Feature Map (4x4):
1 3 2 4
5 6 7 8
3 2 1 0
1 2 3 4

Max Pooling (2x2 window, stride 2):
6 8
3 4
```

---

## Summary Table
| Term           | Description                                                      |
|----------------|------------------------------------------------------------------|
| Convolution    | Feature extraction using a sliding kernel                        |
| Kernel/Filter  | Small matrix of weights applied to the image                     |
| Padding        | Adding extra pixels to control output size                       |
| Same Padding   | Output size = input size                                         |
| Valid Padding  | No padding, output is smaller                                    |
| Stride         | Step size for moving the kernel                                  |
| Channels       | Number of color components (e.g., 1 for grayscale, 3 for RGB)    |
| Max Pooling    | Downsampling by taking the maximum value in each window          |
| Average Pooling| Downsampling by taking the average value in each window          |

---

## References
- [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [DeepLearning.AI Convolutional Neural Networks](https://www.coursera.org/learn/convolutional-neural-networks)
- [Wikipedia: Convolution](https://en.wikipedia.org/wiki/Convolution)
