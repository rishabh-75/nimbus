import numpy as np


class Euler(object):
    def __init__(self, options):
        self.phi = options.get("phi")
        self.theta = options.get("theta")
        self.psi = options.get("psi")

    def radToDeg(self, rad):
        return rad * (180.0 / np.pi)

    def degToRad(self, deg):
        return deg * (np.pi / 180.0)

    def setEulerAnglesToRadians(self):
        self.phi = self.degToRad(self.phi)
        self.theta = self.degToRad(self.theta)
        self.psi = self.degToRad(self.psi)

    def setEulerAnglesToDegrees(self):
        self.phi = self.radToDeg(self.phi)
        self.theta = self.radToDeg(self.theta)
        self.psi = self.radToDeg(self.psi)

    def rotationX(self):
        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(self.phi), np.sin(self.phi)],
                [0, -np.sin(self.phi), np.cos(self.phi)],
            ]
        )

    def rotationY(self):
        return np.array(
            [
                [np.cos(self.theta), 0, -np.sin(self.theta)],
                [0, 1, 0],
                [np.sin(self.theta), 0, np.cos(self.theta)],
            ]
        )

    def rotationZ(self):
        return np.array(
            [
                [np.cos(self.psi), np.sin(self.psi), 0],
                [-np.sin(self.psi), np.cos(self.psi), 0],
                [0, 0, 1],
            ]
        )

    def rotationEulerXYZ(self):
        Rx = self.rotationX()
        Ry = self.rotationY()
        Rz = self.rotationZ()
        return np.matmul(Rx, np.matmul(Ry, Rz))

    def dcmEulerXYZ(self):
        dcm = self.rotationEulerXYZ()
        phi = self.radToDeg(np.arctan2(dcm[1][2], dcm[2][2]))
        theta = self.radToDeg(-np.arcsin(dcm[0][2]))
        psi = self.radToDeg(np.arctan2(dcm[0][1], dcm[0][0]))
        return (phi, theta, psi)


# dcm_options = dict()
# dcm_options.update(phi=-30.0, theta=65.0, psi=-45.0)

# print(
#     "Euler Angles [{}, {}, {}]".format(
#         dcm_options["phi"], dcm_options["theta"], dcm_options["psi"]
#     )
# )
# eulerTransform = Euler(dcm_options)
# eulerTransform.setEulerAnglesToRadians()
# print(eulerTransform.rotationEulerXYZ())
# print(eulerTransform.dcmEulerXYZ())
