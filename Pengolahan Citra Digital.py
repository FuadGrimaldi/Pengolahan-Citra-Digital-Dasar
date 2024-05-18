import sys
import cv2
import math
import imutils #unntuk melakukan perubahan ukuran
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import numpy as np
from matplotlib import pyplot as plt
from skimage import data, exposure
from skimage.feature import hog
import dlib

#membuat class “ShowImage”
class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.image = None
        self.button_load.clicked.connect(self.loadClicked)
        self.button_load_3.clicked.connect(self.reset)
        self.button_load_5.clicked.connect(self.reset_2)
        self.button_load_4.clicked.connect(self.save)
        # Peningkatan Operasi Titik
        self.actionGrayscale.triggered.connect(self.grayClicked)
        self.actionBrightness.triggered.connect(self.brightness)
        self.actionSimple_Kontras.triggered.connect(self.contrast)
        self.actionKontras_Streching.triggered.connect(self.contrastStretching)
        self.actionNegatif_Image.triggered.connect(self.negative)
        self.actionBiner.triggered.connect(self.biner)
        self.actionTranslasi.triggered.connect(self.translasi)
        # Histogram
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.RGBHistogram)
        self.actionHistogram_Equal.triggered.connect(self.equalHistogram)
        # Peningkatan Citra operasi geometri
        self.action45_Derajat.triggered.connect(self.rotasi45derajat)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.action180_Derajat.triggered.connect(self.rotasi180derajat)
        self.action_45_Derajat.triggered.connect(self.rotasiNegatif45derajat)
        self.action_90_Derajat.triggered.connect(self.rotasiNegatif90derajat)
        self.actionTranspose.triggered.connect(self.transpose)
        self.action2x.triggered.connect(self.zoomIn2x)
        self.action3x.triggered.connect(self.zoomIn3x)
        self.action4x.triggered.connect(self.zoomIn4x)
        self.action1_2x.triggered.connect(self.zoomOutsetengah)
        self.action1_4x.triggered.connect(self.zoomOut2)
        self.action3_4x.triggered.connect(self.zoomOut3)
        self.actionCrop.triggered.connect(self.crop)
        # Peningkatan citra operasi aritmatika
        self.actionTambah_Kurang_Kali_dan_Bagi.triggered.connect(self.aritmatika)
        # Peningkatan citra menggunakan operasi logika
        self.actionAND_OR_dan_XOR.triggered.connect(self.operasiLogika)
        # Kernel
        self.actionKernel_A.triggered.connect(self.konvolusiA)
        self.actionKernel_B.triggered.connect(self.konvolusiB)
        self.action2x2.triggered.connect(self.mean2x2)
        self.actionx3.triggered.connect(self.mean3x3)
        self.actionGaussian_Filter.triggered.connect(self.Gaussian)
        self.actionKernel_1.triggered.connect(self.sharpening1)
        self.actionKernel_2.triggered.connect(self.sharpening2)
        self.actionKernel_3.triggered.connect(self.sharpening3)
        self.actionKernel4.triggered.connect(self.sharpening4)
        self.actionKernel_5.triggered.connect(self.sharpening5)
        self.actionKernel_6.triggered.connect(self.sharpening6)
        self.actionLaplace.triggered.connect(self.Laplace)
        self.actionMedian_Filter.triggered.connect(self.Median)
        self.actionMax_Filter.triggered.connect(self.Max)
        self.actionMin_Filter.triggered.connect(self.Min)
        # Edge Detection
        self.actionDFT_Smoothing.triggered.connect(self.dft_smooth)
        self.actionDFT_Edge.triggered.connect(self.dft_edge)
        self.actionSobel.triggered.connect(self.Sobel)
        self.actionCanny.triggered.connect(self.canny)
        self.actionPrewit.triggered.connect(self.Prewitt)
        self.actionRobert.triggered.connect(self.Robert)
        # Morfologi
        self.actionErosi.triggered.connect(self.erosi)
        self.actionDilasi.triggered.connect(self.dilasi)
        self.actionOpening.triggered.connect(self.opening)
        self.actionClosing.triggered.connect(self.closing)
        # Segmentasi
        self.actionBinary.triggered.connect(self.tresh_binary)
        self.actionBinary_Invers.triggered.connect(self.binar_inv)
        self.actionTrunc.triggered.connect(self.trunc)
        self.actionTo_Zero.triggered.connect(self.to_zero)
        self.actionTo_Zero_Invers.triggered.connect(self.to_zero_inv)
        self.actionMean_Thresholding.triggered.connect(self.meanThresholding)
        self.actionGaussian_Thresholding.triggered.connect(self.gaussThresholding)
        self.actionOtsu_Thresholding.triggered.connect(self.otsuThresholding)
        self.actionContour.triggered.connect(self.contour)
        self.actionSkeleton.triggered.connect(self.skeleton)
        # Color Processing
        self.actionTracking.triggered.connect(self.track)
        self.actionPicker.triggered.connect(self.pick)

        # cascade
        self.actionObjek_Detection.triggered.connect(self.objectdetection)
        self.actionHaar_Cascade_Eye_and_Face.triggered.connect(self.FaceandEye)
        self.actionHaar_Cascade_Pedestrian.triggered.connect(self.Pedestrian)
        self.actionCircle_Hough_Transform.triggered.connect(self.CircleHough)
        self.actionHOG.triggered.connect(self.hog)
        self.actionHOG_Jalan.triggered.connect(self.HOGJalan)

        self.actionImage_Landmarks.triggered.connect(self.mask_detected)

    def reset(self):
        self.image = None
        self.label.clear()  # Menghapus konten label
        self.label.setText("No Image")  # Menetapkan teks default
    def reset_2(self):
        self.image = None
        self.label_2.clear()  # Menghapus konten label
        self.label_2.setText("No Image")  # Menetapkan teks default

    def save(self):
        if self.image is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Image", "Output Image", "Images (*.png *.jpg *.jpeg)")
            if file_name:
                cv2.imwrite(file_name, self.image)
                QMessageBox.information(self, "Info", "Foto Berhasil Tersimpan!")
        else:
            QMessageBox.warning(self, "Warning", "Tidak Ada Foto, Gagal Menyimpan!")

    # membuat prosedur button clicked untuk load
    def loadClicked(self):
        # Open a file dialog to select an image file
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image File",
            r"C:\python\python 3.11.5\Code\Pengolahan Citra Digital\Image",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
            options=options
        )

        if file_path:
            self.loaded_image_path = file_path
            # Read the selected image file
            self.image = cv2.imread(file_path)

            if self.image is not None:
                print("Image read successfully")
                # Display the image using the existing method
                self.displayImage(self.image, self.label)
            else:
                print("Failed to read image")
        else:
            print("No file selected")


    # membuat prosedur display image
    def displayImage(self, image, label):
        if self.image is None:
            return

        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        # cv membaca image dalam format BGR, PyQt membaca dalam format RGB
        img = img.rgbSwapped()

        pixmap = QPixmap.fromImage(img)
        label.setPixmap(pixmap)

        self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label.setScaledContents(True)
        self.label_2.setScaledContents(True)


    def grayClicked(self):
        if self.image is not None:
            H, W = self.image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                         0.587 * self.image[i, j, 1] +
                                         0.114 * self.image[i, j, 2], 0, 255)
            self.image = gray
            # Display the grayscale image in the second label (self.label_2)
            self.displayImage(gray, self.label_2)
    def brightness(self):
        # error handling
        try :
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except :
            pass

        H, W = self.image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.image.item(i,j)
                b = np.clip(a + brightness, 0, 255)

                self.image.itemset((i, j), b)
        self.displayImage(self.image, self.label_2)
    #
    def contrast(self):
        # error handling
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.image.itemset((i, j), b)
        self.displayImage(self.image, self.label_2)
    #
    def contrastStretching(self):
        # error handling
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        minV = np.min(self.image)
        maxV = np.max(self.image)

        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.image.itemset((i, j), b)

        self.displayImage(self.image, self.label_2)
    #
    def negative(self):
        # error handling
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]

        for i in range(H):
            for j in range(W):
                pixel_value = self.image.item(i, j)
                negative_value = 255 - pixel_value

                self.image.itemset((i, j), negative_value)
        self.displayImage(self.image, self.label_2)
    #
    def biner(self):
        # Error handling
        try:
            # Convert the image to grayscale if it's not already
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        # Get the dimensions of the image
        H, W = self.image.shape[:2]

        # Iterate through each pixel of the image
        for i in range(H):
            for j in range(W):
                # Get the pixel value at (i, j)
                pixel_value = self.image.item(i, j)
                # If the pixel value is greater than or equal to the threshold, set it to 255 (white), otherwise set it to 0 (black)
                if pixel_value == 180:
                    self.image.itemset((i, j), 0)
                elif pixel_value < 180:
                    self.image.itemset((i, j), 1)
                elif pixel_value > 180:
                    self.image.itemset((i, j), 255)

                else:
                    self.image.itemset((i, j), 0)

        # Display the binary image
        self.displayImage(self.image, self.label_2)

    # A9
    def grayHistogram(self):
        if self.image is not None:
            H, W = self.image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                         0.587 * self.image[i, j, 1] +
                                         0.114 * self.image[i, j, 2], 0, 255)
            self.image = gray
            # Display the grayscale image in the second label (self.label_2)
            self.displayImage(gray, self.label_2)
            # mengubah matrik 2 dimensi menjadi array 1 dimensi
            plt.hist(self.image.ravel(), 255, [0, 255])  # membuat histogram dari pixel
            plt.show()

    # A10
    def RGBHistogram(self):
        color = ("b", "g", "r")  # warna citra
        for i, col in enumerate(color):
            histo = cv2.calcHist([self.image], [i], None, [256],
                                 [0, 256])  # menghitung histogram dari sekumpulan aaray/koleksi
            plt.plot(histo, color=col)  # plotting kepada histogram
            plt.xlim([0, 256])  # mengatur batas sumbu x
        self.displayImage(self.image, self.label_2)
        plt.show()

    # A11
    def equalHistogram(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        # cumulatif distribusi frekuency ( CDF)
        cdf = hist.cumsum()  # cumulative sum (total akumulatif) dari elemen dalam array
        cdf_normalized = cdf * hist.max() / cdf.max()  # CDF dinormalisasi dan dikonversi ke skala 0-255.
        cdf_m = np.ma.masked_equal(cdf, 0)
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
        cdf = np.ma.filled(cdf_m, 0).astype('uint8')
        self.image = cdf[self.image]
        self.displayImage(self.image, self.label_2)
        plt.plot(cdf_normalized, color='b')
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    # B1
    def translasi(self):
        h, w = self.image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]])  # pembuatan matrixs
        img = cv2.warpAffine(self.image, T, (w, h))  # mengubah posisi pixel dalam gambar
        self.image = img
        self.displayImage(self.image, self.label_2)

    # B2
    def rotasi(self, degree):
        h, w = self.image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree,
                                                 .7)  # mendapatkan matriks rotasi dengan parameter degree.
        cos = np.abs(rotationMatrix[0, 0])  # menghitung nilai sin cos dari matrix
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        rotationMatrix[0, 2] += (
                                            nW / 2) - w / 2  # matriks rotasi digunakan untuk menggeser gambar ke tengah setelah diputar
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        rot_image = cv2.warpAffine(self.image, rotationMatrix,
                                   (h, w))  # mengubah posisi pixel dalam gambar secara efektif
        self.image = rot_image
        self.displayImage(self.image, self.label_2)

    def transpose(self):  # orientasi potret menjadi orientasi lanskap
        self.image = cv2.transpose(self.image)
        self.displayImage(self.image, self.label_2)

    def rotasi45derajat(self):
        self.rotasi(45)

    def rotasi90derajat(self):
        self.rotasi(90)

    def rotasi180derajat(self):
        self.rotasi(180)

    def rotasiNegatif45derajat(self):
        self.rotasi(-45)

    def rotasiNegatif90derajat(self):
        self.rotasi(-90)

    # B3
    def zoom(self, skala):
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        # resize_img = cv2.resize(self.image, (900, 400), interpolation=cv2.INTER_AREA) ## (mengatur image berdasarkan dimensi yang ditentukan
        cv2.imshow("Original", self.image)
        cv2.imshow("Zoom", resize_image)
        cv2.waitKey()  # menampilkan gambar

    def zoomIn2x(self):
        self.zoom(2)

    def zoomIn3x(self):
        self.zoom(3)

    def zoomIn4x(self):
        self.zoom(4)

    def zoomOutsetengah(self):
        self.zoom(.5)

    def zoomOut2(self):
        self.zoom(.25)

    def zoomOut3(self):
        self.zoom(.75)

    # B4
    def crop(self):
        h, w = self.image.shape[:2]
        start_row = 40
        start_col = 150
        end_row = 60
        end_col = 300
        crop_image = self.image[start_row:start_col, end_row:end_col]
        # cv2.imshow("Original", self.image)
        # cv2.imshow("Crop", crop_image)
        # cv2.waitKey()
        self.image = crop_image
        self.displayImage(self.image, self.label_2)

    # C1

    def aritmatika(self):
        image1 = cv2.imread("Image/img2.jpg", 0)
        image2 = cv2.imread('Image/img3.jpg', 0)
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        image_kali = image1 * image2
        image_bagi = image1 / image2
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Image Tambah", image_tambah)
        cv2.imshow("Image Kurang", image_kurang)
        cv2.imshow("Image kali", image_kali)
        cv2.imshow("Image bagi", image_bagi)
        cv2.waitKey()

    # C2
    def operasiLogika(self):
        image1 = cv2.imread("Image/img2.jpg", 1)  # dibaca dengan mode RGB
        image2 = cv2.imread('Image/img3.jpg', 1)
        image1 = cv2.cvtColor(image1,
                              cv2.COLOR_BGR2RGB)  # mengubah format warna gambar dari BGR (Blue-Green-Red) ke RGB (Red-Green-Blue).
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasiAnd = cv2.bitwise_and(image1, image2)
        operasiOr = cv2.bitwise_or(image1, image2)
        operasiXor = cv2.bitwise_xor(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Operasi AND", operasiAnd)
        cv2.imshow("Operasi OR", operasiOr)
        cv2.imshow("Operasi XOR", operasiXor)

        cv2.waitKey()

    # D1
    def Konvolusi(self, X, F):  # konvolusi 2d
        X_height = X.shape[0]  # tinggi dan lebar citra
        X_width = X.shape[1]
        F_height = F.shape[0]  # tinggi dan lebar kernel
        F_width = F.shape[1]
        H = (F_height) // 2  # mencari titik tengah kernel
        W = (F_width) // 2
        out = np.zeros((X_height, X_width))
        for i in np.arange(H + 1, X_height - H):  # mengatur pergerakan karnel pada tinggi citra
            for j in np.arange(W + 1, X_width - W):  # mengatur pergerakan kernel pada lebar citra
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)  # menjumlahkan nilai a yang dikali w
                out[i, j] = sum  # menampung hasil yang disimpan sesuai titik koordinat
        return out

    def Konvolusi_Filtering(self, p1, p2, p3, p4, p5, p6, p7, p8, p9):
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)  # mengubah citra menjadi greyscale
        kernel = np.array(
            [
                # array kernel
                [p1, p2, p3],
                [p4, p5, p6],
                [p7, p8, p9]
            ])

        img_output = self.Konvolusi(img, kernel)  # parameter dari fungsi konvolusi 2d, citra dan kernel
        plt.imshow(img_output, cmap='gray', interpolation='bicubic')  # memperhalus tepi citra
        plt.xticks([]), plt.yticks([])
        print('Nilai Pixel Filtering : \n', img_output)
        plt.show()  # Menampilkan gambar

    def konvolusiA(self):
        self.Konvolusi_Filtering(1, 1, 1, 1, 1, 1, 1, 1, 1)

    def konvolusiB(self):
        self.Konvolusi_Filtering(6, 0, -6, 6, 1, -6, 6, 0, -6)

    # D2
    def Mean_filter(self, koefisien, p1, p2, p3, p4, p5, p6, p7, p8, p9):  # mean
        mean = (1.0 / koefisien) * np.array(  # nilai intensitas pixel diganti menjadi rata" filter
            [
                # array kernel 3x3
                [p1, p2, p3],
                [p4, p5, p6],
                [p7, p8, p9]
            ]
        )
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_output = self.Konvolusi(img, mean)
        print('Nilai Pixel Mean Filter: \n', img_output)  # memunculkan nilai pixel pada terminal
        plt.imshow(img_output, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # mengatur lokasi tick dan label sumbu x dan y
        plt.show()

    def mean2x2(self):
        self.Mean_filter(4, 0, 0, 0, 0, 1, 1, 0, 1, 1)

    def mean3x3(self):
        self.Mean_filter(9, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    # D3
    def Gaussian(self):  # fungsi filter mereduksi atau mengeurangi noise
        gausian = (1.0 / 345) * np.array(  # nilai koefisien
            [
                # Kernel gaussian
                [1, 5, 7, 5, 1],
                [5, 20, 33, 20, 5],
                [7, 33, 55, 33, 7],
                [5, 20, 33, 20, 5],
                [1, 5, 7, 5, 1]
            ]
        )
        img = cv2.cvtColor(self.image,
                           cv2.COLOR_BGR2GRAY)  # mengubah gambar dari format warna BGR ke skala keabuan (grayscale)
        img_output = self.Konvolusi(img, gausian)
        print('Nilai Pixel Gaussian: \n', img_output)
        plt.imshow(img_output, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # mengatur lokasi tick dan label sumbu x dan y
        plt.show()

    # D4
    def sharpening_filter(self, p1, p2, p3, p4, p5, p6, p7, p8,
                          p9):  # fungsi meningkatkan kejelasan dan ketajaman detail dalam gambar.
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            # array kernel
            [
                [p1, p2, p3],
                [p4, p5, p6],
                [p7, p8, p9]
            ]
        )
        img_output = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        print('Nilai Pixel Kernel Sharpening: \n', img_output)
        plt.imshow(img_output, cmap='gray', interpolation='bicubic')  # memperhalus tepi citra
        plt.show()
        cv2.waitKey()

    def sharpening1(self):
        self.sharpening_filter(-1, -1, -1, -1, 8, -1, -1, -1, -1)

    def sharpening2(self):
        self.sharpening_filter(-1, -1, -1, -1, 9, -1, -1, -1, -1)

    def sharpening3(self):
        self.sharpening_filter(0, -1, 0, -1, 5, -1, 0, -1, 0)

    def sharpening4(self):
        self.sharpening_filter(1, -2, 1, -2, 5, -2, 1, -2, 1)

    def sharpening5(self):
        self.sharpening_filter(1, -2, 1, -2, 4, -2, 1, -2, 1)

    def sharpening6(self):
        self.sharpening_filter(0, 1, 0, 1, -4, 1, 0, 1, 0)

    def Laplace(self):  # menandai bagian yang menjadi detail citra dan memperbaiki serta mengubah citra
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = (1.0 / 16) * np.array(  # nilai koefisien kernel, array kernel Laplace
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0]
            ])
        img_output = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        plt.imshow(img_output, cmap='gray', interpolation='bicubic')  # memperhalus tepi citra
        print('Nilai Pixel Kernel Sharpening Laplace: \n', img_output)
        plt.title("Laplace")
        plt.show()
        cv2.waitKey()

    # D5
    def Median(self):  # nilai tengah pixel setelah diurutkan
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_output = img.copy()
        H, W = img.shape[:2]  # tinggi dan lebar citra

        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                matrix = []  # menampung nilai pixel
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)  # a digunakan untuk menampung hasil
                        matrix.append(a)  # menambahkan a ke list matrix
                matrix.sort()  # mengurutkan nilai yang ada di list matrix
                median = matrix[24]
                b = median
                img_output.itemset((i, j), b)  # mengganti dengan nilai img output dengan nilai yang baru
        print('Nilai Pixel Median Filter: \n', img_output)
        plt.imshow(img_output, cmap='gray', interpolation='bicubic')
        plt.title("Median Filter")
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Max(self):  # mengurangi dimensi citra atau ukuran citra
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_output = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):  # loop nilai setiap pixel
            for j in np.arange(3, W - 3):
                max = 0
                for k in np.arange(-3, 4):  # mencari nilai maximun
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)  # untuk menampung nilai hasil baca pixel
                        if a > max:
                            max = a
                        b = max
                img_output.itemset((i, j), b)
        print('Nilai Pixel Maksimal Filter :\n', img_output)
        plt.title("Max Filter")
        plt.imshow(img_output, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Min(self):  # mengganti nilai pixel menjadi yang pixel minimum untuk mengurangi dimensi citra
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_output = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):  # loop nilai setiap pixel
            for j in np.arange(3, W - 3):
                min = 0
                for k in np.arange(-3, 4):  # mencari nilai maximun
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)  # untuk menampung nilai hasil baca pixel
                        if a < min:
                            min = a
                        b = min
                img_output.itemset((i, j), b)
        print('Nilai Pixel Minimun Filter: \n', img_output)  # memunculkan nilai pixel pada terminal
        plt.imshow(img_output, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # E1
    def dft_smooth(self):
        # menghasilkan gelombang sinus dengan frekuensi 3 dan menambahkannya ke nilai maksimumnya.
        x = np.arange(256)
        y = np.sin(2*np.pi*x/3)
        y += max(y)
        # menggunakan gelombang sinus yang telah dihasilkan sebelumnya dan memetakkannya ke dalam skala abu - abu.
        img = np.array([[y[j]*127 for j in range(256)] for i in range(256)], dtype=np.uint8)
        plt.imshow(img)
        img = cv2.imread('Image/img3.jpg', 0)

        # Menggeser frekuensi rendah ke tengah untuk visualisasi yang lebih baik dan Menghitung spektrum magnitudo
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])))

        rows, cols = img.shape
        crow, ccol = int(rows/2), int(cols/2)
        # membuat mask lingkaran dengan radius 50 untuk digunakan pada spektrum
        mask = np.zeros((rows, cols, 2), np.uint8)
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r)
        mask[mask_area] = 1
        # perkalian antara spektrum Fourier, shift(dft_shift) dengan masker lingkaran(mask)
        fshift = dft_shift*mask
        # menghitung magnitudo spektrum hasilnya.
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))

        # Menggeser kembali frekuensi rendah ke posisi semula, dan melakukan invers Fourier untuk mendapatkan citra kembali
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

        # Menampilkan citra asli, spektrum frekuensi citra, hasil dari aplikasi mask pada spektrum, dan citra hasil invers Fourier
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')

        plt.show()
    # E2
    def dft_edge(self):
        x = np.arange(256)
        # sinyal gelombang sinus dengan frekuensi 3 (dibagi oleh 3 dalam rumus), dan memetakan nilainya pada array x.
        y = np.sin(2*np.pi*x/3)
        y += max(y)

        # menggunakan gelombang sinus yang telah dihasilkan sebelumnya dan memetakkannya ke dalam skala abu - abu.
        img = np.array([[y[j]*127 for j in range(256)] for i in range(256)], dtype=np.uint8)
        plt.imshow(img)
        img = cv2.imread('Image/img3.jpg', 0)
        # menggeser hasilnya sehingga frekuensi rendah berada di tengah.
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Menghitung spektrum magnitudo
        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])))

        # bentuk citra, kolom dan baris
        rows, cols = img.shape
        crow, ccol = int(rows/2), int(cols/2)
        # fungsi array berisi nilai 1, nilai 2 bagian real dan imajiner dari spektrum
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 70
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r)
        mask[mask_area] = 0
        fshift = dft_shift*mask
        # menghitung magnitudo spektrum hasilnya.
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))

        # Menggeser kembali frekuensi rendah ke posisi semula, dan melakukan invers Fourier untuk mendapatkan citra kembali
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

        # plotting
        fig = plt.figure(figsize=(12, 12))
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')

        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_back, cmap='gray')
        ax4.title.set_text('Inverse Fourier')

        plt.show()
    # F1
    def turunan_pertama(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Inisialisasi kernel sumbu X
        kernel_x = np.array([[x1, x2, x3], [x4, x5, x6], [x7, x8, x9]])

        # Inisialisasi kernel sumbu Y
        kernel_y = np.array([[y1, y2, y3], [y4, y5, y6], [y7, y8, y9]])

        # Konvolusi citra terhadap kernel sumbu X dan Y
        Gx = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        Gy = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

        # Hitung Gradien
        gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        # Normalisasi panjang gradien dalam range 0-255
        gradient_normalized = ((gradient_magnitude / gradient_magnitude.max()) * 255).astype(np.uint8)

        # Menampilkan citra keluaran dalam color map 'gray' dan dengan interpolasi 'bicubic'
        plt.imshow(gradient_normalized, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()

    def Sobel(self):
        self.turunan_pertama(-1, 0, 1, -2, 0, 2, -1, 0, 1, -1, -2, -1, 0, 0, 0, 1, 2, 1)
    def Prewitt(self):
        # Kernel Prewitt x
        # [
        #     [-1, 0, 1],
        #     [-1, 0, 1],
        #     [-1, 0, 1]
        # ]
        # Kernel Prewitt y
        # [
        #     [-1, -1, -1],
        #     [0, 0, 0],
        #     [1, 1, 1]
        # ]

        self.turunan_pertama(-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1)

    def Robert(self):
        # Kernel Robert y
        # [
        #     [-1, 0, 0],
        #     [0, 1, 0],
        #     [0, 0, 0]
        # ]
        #
        # Kernel Robert x
        # [
        #     [0, 0, -1],
        #     [0, 1, 0],
        #     [0, 0, 0]
        # ]
        self.turunan_pertama(0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0)
    # F2
    def canny(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        conv = (1 / 345) * np.array(
            [[1, 5, 7, 5, 1],
             [5, 20, 33, 20, 5],
             [7, 33, 55, 33, 7],
             [5, 20, 33, 20, 5],
             [1, 5, 7, 5, 1]])

        out_img = self.Konvolusi(img, conv)
        out_img = out_img.astype("uint8")
        cv2.imshow("Noise reduction", out_img)

        # finding gradient
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        img_x = self.Konvolusi(out_img, Sx)
        img_y = self.Konvolusi(out_img, Sy)
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        img_out = (img_out / np.max(img_out)) * 255
        cv2.imshow("finding Gradien", img_out)
        theta = np.arctan2(img_y, img_x)

        # non-maximum suppression
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180
        H, W = img.shape[:2]
        Z = np.zeros((H, W), dtype=np.int32)
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img_out[i, j + 1]
                        r = img_out[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i - 1, j - 1]
                        r = img_out[i + 1, j + 1]
                    if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                        Z[i, j] = img_out[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass
        img_N = Z.astype("uint8")
        cv2.imshow("Non Maximum Suppression", img_N)
        # Eliminasi titik tepi lemah jika tidak terhubung dengan tetangga tepi kuat
        # hysteresis thresholding part 1
        weak = 15
        strong = 90
        for i in np.arange(H):
            for j in np.arange(W):
                a = img_N.item(i, j)
                if (a > weak):
                    b = weak
                    if (a > strong):
                        b = 255
                else:
                    b = 0
                img_N.itemset((i, j), b)
        img_H1 = img_N.astype("uint8")
        cv2.imshow("hysteresis part 1", img_H1)
        # hysteresis thresholding part 2
        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or (
                                img_H1[i + 1, j + 1] == strong) or (img_H1[i, j - 1] == strong) or (
                                img_H1[i, j + 1] == strong) or (img_H1[i - 1, j - 1] == strong) or (
                                img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass
        img_H2 = img_H1.astype("uint8")
        cv2.imshow("hysteresis part2", img_H2)

# H1 Segmentasi citra (morfologi)
    def morfologi_biner(self):
        # Konversi citra ke citra grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Konversi citra ke citra biner menggunakan thresholding
        thresh, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # Inisialisasi kernel untuk operasi morfologi
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        return binary_image, kernel

    def erosi(self):
        binary_image, kernel = self.morfologi_biner()
        eroded = cv2.erode(binary_image, kernel)
        # Menampilkan citra hasil operasi morfologi
        cv2.imshow('Eroded Image', eroded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def dilasi(self):
        binary_image, kernel = self.morfologi_biner()
        dilated = cv2.dilate(binary_image, kernel)
        # Menampilkan citra hasil operasi morfologi
        cv2.imshow('Dilated Image', dilated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def opening(self):
        binary_image, kernel = self.morfologi_biner()
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        # Menampilkan citra hasil operasi morfologi
        cv2.imshow('Opening Image', opening)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def closing(self):
        binary_image, kernel = self.morfologi_biner()
        closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        # Menampilkan citra hasil operasi morfologi
        cv2.imshow('Closing Image', closing)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def skeleton(self):
        binary_image, kernel = self.morfologi_biner()
        skel = np.zeros(binary_image.shape, np.uint8)
        element = np.array([[1, 0, 0],
                            [0, 1, 1],
                            [0, 1, 1]], dtype=np.uint8)

        while True:
            eroded = cv2.erode(binary_image, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(binary_image, temp)
            skel = cv2.bitwise_or(skel, temp)
            binary_image = eroded.copy()

            if cv2.countNonZero(binary_image) == 0:
                break

        cv2.imshow('Skeletonized Image', skel)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # # H1 Segmentasi citra (Global threshold)
    def tresh(self, threshold_type):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        I = 127
        maxval = 255
        _, thresh = cv2.threshold(gray, I, maxval, threshold_type)
        return thresh
    def tresh_binary(self):
        thresh = self.tresh(cv2.THRESH_BINARY)
        cv2.imshow('Binary', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def binar_inv(self):
        thresh = self.tresh(cv2.THRESH_BINARY_INV)
        cv2.imshow('Binary Invers', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def trunc(self):
        thresh = self.tresh(cv2.THRESH_TRUNC)
        cv2.imshow('Trunc', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def to_zero(self):
        thresh = self.tresh(cv2.THRESH_TOZERO)
        cv2.imshow('To Zero', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def to_zero_inv(self):
        thresh = self.tresh(cv2.THRESH_TOZERO_INV)
        cv2.imshow('To Zero Invers', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # H2 Segmentasi citra (Local threshold)
    def meanThresholding(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print("Pixel Awal", gray)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        print("Pixel mean thresholding", thresh)
        cv2.imshow('Mean Thresholding', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def gaussThresholding(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print("Pixel Awal", gray)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        print("Pixel Gaussian thresholding", thresh)
        cv2.imshow('Gaussian Thresholding', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def otsuThresholding(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        I = 127
        _, thresh = cv2.threshold(gray, I,  255,  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        print("Pixel otsu thresholding", thresh)
        cv2.imshow('Otsu Thresholding', thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # H3 Segmentasi citra (Contour)
    def contour(self):
        self.image = cv2.imread('Image/img5.jpg')
        # Konversi citra RGB ke grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Threshold citra dengan nilai T=127
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Ekstrak kontur
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Mendapatkan poligon pendekatan dan titik tengah kontur untuk setiap kontur
        approximate_polygons = []
        contour_centers = []
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approximate_polygons.append(approx)
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contour_centers.append((cX, cY))

        # Gambar kontur pada citra
        for contour in contours:
            cv2.drawContours(self.image, [contour], -1, (0, 0, 0), 2)

        # Memeriksa jenis poligon
        for i, polygon in enumerate(approximate_polygons):
            num_sides = len(polygon)
            center = contour_centers[i]
            shape = "Undefined"
            if num_sides == 3:
                shape = "Triangle"
            elif num_sides == 4:
                # Memeriksa jika poligon dengan 4 sisi, apakah persegi atau persegi panjang
                x, y, w, h = cv2.boundingRect(polygon)
                aspect_ratio = float(w) / h
                shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif num_sides == 5:
                shape = "Pentagon"
            elif num_sides == 6:
                shape = "Hexagon"
            else:
                shape = "Circle"
            # Menampilkan hasil deteksi pada citra
            cv2.putText(self.image, shape, (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                        2)
            cv2.circle(self.image, center, 3, (0, 0, 0), -1)


        # Menampilkan citra dengan hasil deteksi
        cv2.imshow("Detected Shapes", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def track(self):
        # Fungsi untuk melacak warna dalam video dari webcam
        cam = cv2.VideoCapture(0)  # Membuka kamera
        while True:
            _, frame = cam.read()  # Membaca frame dari kamera
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Mengonversi gambar ke ruang warna HSV
            lower_color = np.array([0, 0, 0])  # Batas bawah rentang warna yang ingin dilacak (hitam)
            upper_color = np.array([255, 255, 140])  # Batas atas rentang warna yang ingin dilacak (kuning muda)
            mask = cv2.inRange(hsv, lower_color,
                               upper_color)  # Membuat masker untuk memisahkan warna yang ingin dilacak
            result = cv2.bitwise_and(frame, frame, mask=mask)  # Menggabungkan frame asli dengan hasil masker
            cv2.imshow("Frame", frame)  # Menampilkan frame asli
            cv2.imshow("Mask", mask)  # Menampilkan masker warna yang dibuat
            cv2.imshow("Result", result)  # Menampilkan hasil pelacakan warna
            key = cv2.waitKey(1)
            if key == 27:  # Ketika tombol Esc ditekan, keluar dari loop
                break
        cam.release()  # Melepaskan kamera setelah selesai
        cv2.destroyAllWindows()  # Menutup semua jendela tampilan

    def pick(self):
        # Fungsi untuk memilih rentang warna yang ingin dilacak menggunakan trackbar
        def nothing(x):
            pass

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Membuka kamera
        cv2.namedWindow("Trackbar")  # Membuat jendela untuk trackbar
        cv2.createTrackbar("L-H", "Trackbar", 0, 179, nothing)  # Trackbar untuk nilai H rendah
        cv2.createTrackbar("L-S", "Trackbar", 0, 255, nothing)  # Trackbar untuk nilai S rendah
        cv2.createTrackbar("L-V", "Trackbar", 0, 255, nothing)  # Trackbar untuk nilai V rendah
        cv2.createTrackbar("U-H", "Trackbar", 179, 179, nothing)  # Trackbar untuk nilai H tinggi
        cv2.createTrackbar("U-S", "Trackbar", 255, 255, nothing)  # Trackbar untuk nilai S tinggi
        cv2.createTrackbar("U-V", "Trackbar", 255, 255, nothing)  # Trackbar untuk nilai V tinggi

        while True:
            _, frame = cam.read()  # Membaca frame dari kamera
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Mengonversi gambar ke ruang warna HSV

            # Mendapatkan nilai trackbar saat ini
            L_H = cv2.getTrackbarPos("L-H", "Trackbar")
            L_S = cv2.getTrackbarPos("L-S", "Trackbar")
            L_V = cv2.getTrackbarPos("L-V", "Trackbar")
            U_H = cv2.getTrackbarPos("U-H", "Trackbar")
            U_S = cv2.getTrackbarPos("U-S", "Trackbar")
            U_V = cv2.getTrackbarPos("U-V", "Trackbar")

            # Membuat batas bawah dan atas berdasarkan nilai trackbar saat ini
            lower_color = np.array([L_H, L_S, L_V])
            upper_color = np.array([U_H, U_S, U_V])

            mask = cv2.inRange(hsv, lower_color,
                               upper_color)  # Membuat masker untuk memisahkan warna yang ingin dilacak
            result = cv2.bitwise_and(frame, frame, mask=mask)  # Menggabungkan frame asli dengan hasil masker
            cv2.imshow("Frame", frame)  # Menampilkan frame asli
            cv2.imshow("Mask", mask)  # Menampilkan masker warna yang dibuat
            cv2.imshow("Result", result)  # Menampilkan hasil pelacakan warna

            key = cv2.waitKey(1)
            if key == 27:  # Ketika tombol Esc ditekan, keluar dari loop
                break
        cam.release()  # Melepaskan kamera setelah selesai
        cv2.destroyAllWindows()  # Menutup semua jendela tampilan
# I3
    def objectdetection(self):
        cam = cv2.VideoCapture('Video/cctv3.mp4')  # Membaca file video dan menginisialisasi camera
        car_cascade = cv2.CascadeClassifier('HaarCascade/car.xml')  # Memuat file klasifikasi yang berisi deskripsi fitur mobil
        while True:  # Looping video
            ret, frame = cam.read()  # Membaca setiap frame dari video
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah gambar ke dalam grayscale
            cars = car_cascade.detectMultiScale(gray, 1.1, 3)  # Mencari mobil pada frame dengan detektor mobil
            for (x, y, w, h) in cars:  # Loop setiap mobil yang terdeteksi dengan detektor mobil
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),
                              1)  # Mencari mobil pada frame dengan detektor mobil
            cv2.imshow('Video', frame)  # Menampilkan frame mobil yang terdeteksi

            if cv2.waitKey(10) & 0xFF == ord('q'):  # Menekan tombol q untuk menghentikan loop
                break
        cam.release()
        cv2.destroyAllWindows()
    # I4
    def hog(self):
        image = data.astronaut()
        # Ekstraksi fitur Histogram of Oriented Gradients (HOG) dari gambar
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1), visualize=True, channel_axis=-1)
        # Membuat plot untuk menampilkan gambar asli dan HOG
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        # Plot gambar asli
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram untuk tampilan yang lebih baik
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        # Plot HOG
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()
    def HOGJalan(self):
        hog = cv2.HOGDescriptor()  # Membuat objek detektor HOG
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) # menentukan SVM untuk detektor default
        Photo = cv2.imread("Image/pict1.jpg")  # Membaca citra
        Photo = imutils.resize(Photo, width=min(400, Photo.shape[0]))
        # Melakukan resizing citra untuk mempercepat proses deteksi dengan memastikan lebar tidak >400 pixel

        (regions, _) = hog.detectMultiScale(Photo, winStride=(4, 4), padding=(4, 4), scale=1.05)
        # Melakukan deteksi pejalan kaki pada citra menggunakan detektor HOG
        for (x, y, w, h) in regions:  # Looping pada setiap region pejalan kaki yang terdeteksi pada citra
            cv2.rectangle(Photo, (x, y), (x + w, y + h), (0, 0, 255),
                          2)  # Menggambar kotak pada setiap region pejalan kaki yang terdeteksi pada citra
        cv2.imshow("image", Photo)  # Menampilkan citra dengan kotak-kotak deteksi pejalan kaki
        cv2.waitKey()

    # I5
    def FaceandEye(self):
        # Load the face cascade classifier
        face_classifier = cv2.CascadeClassifier('HaarCascade/haarcascade_frontalface_default.xml')
        # Read the image
        Image = cv2.imread('Image/pict2.jpg')
        # Check if the image is loaded successfully
        if Image is None:
            print("Error: Unable to load image")
        else:
            # Convert the image to grayscale
            gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
            # Detect faces in the grayscale image
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            # Check if no faces are detected
            if len(faces) == 0:
                print("No faces found")
            else:
                # Draw rectangles around the detected faces
                for (x, y, w, h) in faces:
                    cv2.rectangle(Image, (x, y), (x + w, y + h), (127, 0, 255), 2)

                # Display the image with rectangles around the faces
                cv2.imshow('Face Detection', Image)
                cv2.waitKey(0)

        # Close all OpenCV windows
        cv2.destroyAllWindows()
    # I6
    def Pedestrian(self):
        body_classifier = cv2.CascadeClassifier('HaarCascade/haarcascade_fullbody.xml')  #  data dengan menggunakan file xml
        cap = cv2.VideoCapture('Video/cctv2.mp4')  # Membaca video
        while cap.isOpened():  # Looping setiap frame
            ret, frame = cap.read()  # Mengambil setiapframe video
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            # resize ukuran frame agar menjadi lebih kecil dengan skala 0.5
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # mengubah frame menjadi gray
            bodies = body_classifier.detectMultiScale(gray, 1.2, 3)  # Menerapkan algoritma deteksi tubuh manusia
            for (x, y, w, h) in bodies: # membuat rectangle pada objek
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255),2)
                cv2.imshow('Pedestrians', frame)
            if cv2.waitKey(1) == ord('q'):  # Menekan tombol q untuk menghentikan loop
                break

        cap.release()
        cv2.destroyAllWindows()
    # I7
    def CircleHough(self):
        img = cv2.imread('Image/pict3.png', 0)  # baca image
        img = cv2.medianBlur(img, 5)  # median filtering untuk memgurangi noise
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # menjadi grayscale
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0,
                                   maxRadius=0)  # Melakukan transformasi Hough lingkaran untuk mendeteksi lingkaran pada gambar
        circles = np.uint16(np.around(circles))  # Mengkonversi nilai koordinat lingkaran menjadi bilangan bulat (integer)
        for i in circles[0, :]:  # Looping
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Menggambar lingkaran
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)  # Menggambar titik
        cv2.imshow('detected circles', cimg)  # Menampilkan gambar berwarna
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image(self, image_path, output_path='image_landmarks.jpg'):
        # Load the image
        image = cv2.imread(image_path)

        # Load the dlib face predictor and detector
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        detector = dlib.get_frontal_face_detector()

        # Detect faces
        rects = detector(image, 1)
        if len(rects) > 1:
            print("Error: Too many faces detected in the image.")
            return
        if len(rects) == 0:
            print("Error: No faces detected in the image.")
            return

        # Get landmarks
        landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

        # Annotate landmarks
        image_with_landmarks = image.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(image_with_landmarks, str(idx), pos, fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, fontScale=0.4,
                        color=(0, 0, 255))
            cv2.circle(image_with_landmarks, pos, 3, color=(0, 255, 255))

        # Display the result
        cv2.imshow('Image with Landmarks', image_with_landmarks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Save the result
        cv2.imwrite(output_path, image_with_landmarks)
    def mask_detected(self):
        self.process_image('Image/faceandeye.JPG')
        # if self.image is not None:
        #     self.process_image(self.image)
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle("Pengolahan Citra Digital Dasar")
    window.show()
    sys.exit(app.exec_())
