from annotated_types import LowerCase
import numpy as np


class Euler(object):
    def __init__(self, eulerAngles):
        self.eulerAngles = eulerAngles

    def radToDeg(self, rad):
        return rad * (180.0 / np.pi)

    def degToRad(self, deg):
        return deg * (np.pi / 180.0)

    def setEulerAnglesToRadians(self):
        self.eulerAngles = [self.degToRad(theta) for theta in self.eulerAngles]

    def setEulerAnglesToDegrees(self):
        self.eulerAngles = [self.radToDeg(theta) for theta in self.eulerAngles]

    def rotationX(self, theta):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), np.sin(theta)],
                [0, -np.sin(theta), np.cos(theta)],
            ]
        )

    def rotationY(self, theta):
        return np.array(
            [
                [np.cos(theta), 0, -np.sin(theta)],
                [0, 1, 0],
                [np.sin(theta), 0, np.cos(theta)],
            ]
        )

    def rotationZ(self, theta):
        return np.array(
            [
                [np.cos(theta), np.sin(theta), 0],
                [-np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def rotationEulerTransformation(self, rotationSequence):
        """
        XYZ: Roll | Pitch | Yaw \n
        ZXZ: Spin | Nutation | Precession \n

        !!! rotation right to left
        """
        try:
            if rotationSequence == "XYZ":
                r3 = self.rotationZ(self.eulerAngles[2])
                r2 = self.rotationY(self.eulerAngles[1])
                r1 = self.rotationX(self.eulerAngles[0])

            if rotationSequence == "ZXZ":
                r3 = self.rotationZ(self.eulerAngles[2])
                r2 = self.rotationX(self.eulerAngles[1])
                r1 = self.rotationZ(self.eulerAngles[0])

            return np.matmul(r1, np.matmul(r2, r3))
        except:
            print("! invalid sequence")

    def dcmEulerTransformation(self, rotationSequence):
        try:
            dcm = self.rotationEulerTransformation(rotationSequence)
            if rotationSequence == "XYZ":
                phi = np.arctan2(dcm[1][2], dcm[2][2])
                theta = -np.arcsin(dcm[0][2])
                psi = np.arctan2(dcm[0][1], dcm[0][0])

            if rotationSequence == "ZXZ":
                phi = np.arctan2(dcm[0][2], dcm[1][2])
                theta = np.arccos(dcm[2][2])
                psi = np.arctan2(dcm[2][0], -dcm[2][1])

            return (phi, theta, psi)
        except:
            print("! invalid sequence")


eulerAngles = [30.0, 65.0, -45.0]
print(
    "Euler Angles [{}, {}, {}]".format(
        eulerAngles[0],
        eulerAngles[1],
        eulerAngles[2],
    )
)
eulerTransform = Euler(eulerAngles)
eulerTransform.setEulerAnglesToRadians()

rotationSequence = "XYZ"
print(
    f"R{rotationSequence.lower()}: \n",
    eulerTransform.rotationEulerTransformation(rotationSequence),
)
print(
    f"Recovered {rotationSequence} Euler Angles: ",
    [
        eulerTransform.radToDeg(theta)
        for theta in eulerTransform.dcmEulerTransformation(rotationSequence)
    ],
)

rotationSequence = "ZXZ"
print(
    f"R{rotationSequence.lower()}: \n",
    eulerTransform.rotationEulerTransformation(rotationSequence),
)
print(
    f"Recovered {rotationSequence} Euler Angles: ",
    [
        eulerTransform.radToDeg(theta)
        for theta in eulerTransform.dcmEulerTransformation(rotationSequence)
    ],
)
