""" QtImageViewer.py: PyQt image viewer widget for a QPixmap in a QGraphicsView scene with mouse zooming and panning.
"""

import sys


from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt, QRectF, QT_VERSION_STR
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QPushButton


class QtImageViewer(QGraphicsView):
    def __init__(self, ):
        QGraphicsView.__init__(self)

        # Image is displayed as a QPixmap in a QGraphicsScene attached to this QGraphicsView.
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # Store a local handle to the scene's current image pixmap.
        self._pixmap_handle = None

        # Image aspect ratio mode.
        # !!! ONLY applies to full image. Aspect ratio is always ignored when zooming.
        #   Qt.IgnoreAspectRatio: Scale image to fit viewport.
        #   Qt.KeepAspectRatio: Scale image to fit inside viewport, preserving aspect ratio.
        #   Qt.KeepAspectRatioByExpanding: Scale image to fill the viewport, preserving aspect ratio.
        self.aspectRatioMode = Qt.KeepAspectRatio

        ok_button = QPushButton('OK', self)
        ok_button.setToolTip('Accept this viewpoint')
        ok_button.move(100, 100)
        ok_button.clicked.connect(self.ok_clicked)

        ok_button = QPushButton('Not OK', self)
        ok_button.setToolTip('Does not accept this viewpoint')
        ok_button.move(200, 100)
        ok_button.clicked.connect(self.not_ok_clicked)

    def ok_clicked(self):
        pass

    def not_ok_clicked(self):
        pass

    def has_image(self):
        """ Returns whether or not the scene contains an image pixmap.
        """
        return self._pixmap_handle is not None

    def clear_image(self):
        """ Removes the current image pixmap from the scene if it exists.
        """
        if self.has_image():
            self.scene.removeItem(self._pixmap_handle)
            self._pixmap_handle = None

    def pixmap(self):
        """ Returns the scene's current image pixmap as a QPixmap, or else None if no image exists.
        :rtype: QPixmap | None
        """
        if self.has_image():
            return self._pixmap_handle.pixmap()
        return None

    def set_pixmap(self, image):
        """ Set the scene's current image pixmap to the input QImage or QPixmap.
        Raises a RuntimeError if the input image has type other than QImage or QPixmap.
        :type image: QImage | QPixmap
        """
        if type(image) is QPixmap:
            pixmap = image
        elif type(image) is QImage:
            pixmap = QPixmap.fromImage(image)
        else:
            raise RuntimeError("ImageViewer.setImage: Argument must be a QImage or QPixmap.")

        if self.has_image():
            self._pixmap_handle.setPixmap(pixmap)
        else:
            self._pixmap_handle = self.scene.addPixmap(pixmap)

        self.setSceneRect(QRectF(pixmap.rect()))  # Set scene size to image size.

    def set_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = channel * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.set_pixmap(q_image)


if __name__ == '__main__':
    print('Using Qt ' + QT_VERSION_STR)

    # Create the application.
    app = QApplication(sys.argv)

    import cv2
    cv_image = cv2.imread('/Users/anhtruong/Desktop/in.jpg')

    # Create image viewer and load an image file to display.
    viewer = QtImageViewer()
    viewer.set_image(cv_image)

    # Show viewer and run application.
    viewer.show()
    sys.exit(app.exec_())
