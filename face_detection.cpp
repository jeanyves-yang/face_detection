#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv )
{
    if ( argc != 2 )
        {
            printf("usage: DisplayImage.out <Image_Path>\n");
            return -1;
        }

        Mat image;
        image = imread( argv[1], 1 );

        if ( !image.data )
        {
            printf("No image data \n");
            return -1;
        }

        int rows = image.rows;
        int cols = image.cols;

        resize(image, image, Size(), 0.05, 0.05, INTER_LINEAR);
        transpose(image,image);
        flip(image,image,cols/2);
        /*rows,cols,z = img.shape
        img = cv2.transpose(img)
        img = cv2.flip(img, cols/2)*/


        namedWindow("Display Image", WINDOW_AUTOSIZE );
        imshow("Display Image", image);

        waitKey(0);
    return 0;
}
