import vtk
import numpy as np
from typing import Union
import enum

class PathlineType(enum.Enum):
    PATHLINE_TUBE = 0
    PATHLINE_LINE = 1

class mvDisplayObject:
    def __init__(self):
        self.mapper = None
        self.actor = vtk.vtkActor()
        #self.actor.VisibilityOff()
        self.lod_actor = vtk.vtkLODActor()
        #self.lod_actor.VisibilityOff()
        self.active_actor = self.actor

    def GetActor(self):
        return self.active_actor

    def _SetMapperInput(self, input: Union[vtk.vtkPolyData, vtk.vtkDataSet]):
        if not isinstance(input, (vtk.vtkPolyData, vtk.vtkDataSet)):
            raise TypeError("Argument must be a vtk.vtkPolyData or a vtk.vtkDataSet")
        if isinstance(input, vtk.vtkPolyData):
            self.mapper = vtk.vtkPolyDataMapper()
            self.actor.SetMapper(self.mapper)
        elif isinstance(input, vtk.vtkDataSet):
            self.mapper = vtk.vtkDataSetMapper()
            self.actor.SetMapper(self.mapper)

class mvPathlines(mvDisplayObject):
    def __init__(self, pathlineFile, asTubes=True):
        super().__init__()
        # mapper is always vtkPolyDataMapper
        self.mapper = vtk.vtkPolyDataMapper()
        self.actor.SetMapper(self.mapper)

        self.pathlineFile = pathlineFile
        self.representation = PathlineType.PATHLINE_TUBE if asTubes else PathlineType.PATHLINE_LINE
        self.m_DefaultTubeDiameter = 1

        (self.m_PathlineDataSet, self.m_PathlineCoordinates, self.m_PathlineScalarArray) = ConvertToPolyData(pathlineFile)

        self.m_Tube = vtk.vtkTubeFilter()
        self.m_Tube.SetNumberOfSides(10)
        self.m_Tube.SetInputData(self.m_PathlineDataSet)
        self._build_pipe_line()

    def SetDefaultTubeDiameter(self, d):
        self.m_Tube.SetRadius(self.GetNormalizedTubeDiameter() * d / 2.0)
        self.m_DefaultTubeDiameter = d

    def SetRepresentationToTube(self):
        self.representation = PathlineType.PATHLINE_TUBE
        self._build_pipe_line()

    def SetRepresentationToLine(self):
        self.representation = PathlineType.PATHLINE_LINE
        self._build_pipe_line()

    def GetRepresentation(self):
        return self.pathline_type
    
    def SetNormalizedTubeDiameter(self, dn):
        self.m_Tube.SetRadius(dn * self.m_DefaultTubeDiameter / 2.0)

    def GetNormalizedTubeDiameter(self):
        return (2 * self.m_Tube.GetRadius() / self.m_DefaultTubeDiameter)

    def _build_pipe_line(self):
        if (self.representation == PathlineType.PATHLINE_TUBE):
            self.mapper.SetInputConnection(self.m_Tube.GetOutputPort())
        else:
            self.mapper.SetInputData(self.m_PathlineDataSet)

'''
based on:
static void ReadData(char *pathlineFile, int &numPathlines, int &numCoordinates,
                     double *&coordinates, double *&scalarArrayTime, double *&scalarArrayMaxTime,
                     double *&scalarArrayMinTime, vtkIdType *&pointArray, bool backwards, double &minPositiveTime);

removed scalarArrayMaxTime, scalarArrayMinTime, minPositiveTime
'''
class ModpathReader:
    @staticmethod
    def ReadData(pathlineFile):
        with open(pathlineFile, 'r') as f:
            line = f.readline()             # Line 1
            if (line[22:23] == "6"):
                version = 6
                line = f.readline()         # Line 2
                if (line[1:2] == "1"):
                    backwards = False
                elif (line[1:2] == "2"):
                    backwards = True
                parts = line.split()
                Number = parts[1]
                tref = float(Number)
                line = f.readline()         # Line 3 END HEADER
                line = f.readline()         # Line 4 1st line of path data
            else:
                assert False, "Only modpath 6 is supported"

            intArray = vtk.vtkIntArray()
            doubleArray = vtk.vtkDoubleArray()

            doubleArray.SetNumberOfComponents(4)
            nlines = 0

            while (True):
                if (version <= 5):
                    assert False, "Only modpath 6 is supported"
                else:
                    parts = line.split()
                    i = int(parts[0])
                    v = [float(parts[5]), float(parts[6]), float(parts[7]), float(parts[4])]

                line = f.readline()
                if len(line) == 0:
                    break

                intCount = intArray.GetNumberOfTuples()
                flag = 0
                w = [0.0, 0.0, 0.0, 0.0]
                for j in range(intCount - 1, -1, -1):
                    if (flag != 0):
                        break
                    if (intArray.GetValue(j) + 1) == i:
                        doubleArray.GetTypedTuple(j, w)
                        if (w[:3] == v[:3]):
                            flag = -1  ## ignore coincident point
                        else:
                            flag = 1   ## different point
                        break
                if (flag != -1):
                    intArray.InsertNextValue(i - 1)
                    doubleArray.InsertNextTuple(v)
                    if (i > nlines):
                        nlines = i

        ncoord = intArray.GetNumberOfTuples()
        size = np.zeros(nlines, dtype=int)
        locator = np.zeros(nlines, dtype=int)
        pointArray = np.zeros(ncoord + nlines, dtype=np.int64)
        coordinates = np.zeros(3 * ncoord, dtype=float)
        scalarArrayTime = np.zeros(ncoord, dtype=float)

        for j in range(ncoord):
            size[intArray.GetValue(j)] += 1

        # locator[i] = position to insert next pathline point index for
        # pathline i.
        numPathlines = 0
        pos = 0
        for i in range(nlines):
            if (size[i] > 1):
                pointArray[pos] = size[i]
                locator[i] = pos + 1
                pos += size[i] + 1
                numPathlines += 1
            else:
                locator[i] = -1   # exclude this line because it has less than 2 points

        k = 0
        if (numPathlines > 0):
            for j in range(ncoord):
                i = intArray.GetValue(j)
                if (locator[i] != -1):
                    w = [0.0] * 4
                    doubleArray.GetTypedTuple(j, w)
                    coordinates[3 * k]     = w[0]
                    coordinates[3 * k + 1] = w[1]
                    coordinates[3 * k + 2] = w[2]

                    if (backwards):
                        scalarArrayTime[k] = tref - w[3]
                    else:
                        scalarArrayTime[k] = tref + w[3]
                    pointArray[locator[i]] = k
                    locator[i] += 1
                    k += 1

        numCoordinates = k
        return (numPathlines, numCoordinates, coordinates, scalarArrayTime, pointArray, backwards)

def ConvertToPolyData(pathlineFile):

    (m_NumberOfPathlines, m_NumberOfPathlineCoordinates,  m_PathlineCoordinates, m_PathlineScalarArray, m_PathlinePointArray, m_PathsBackwardsInTime) = ModpathReader.ReadData(pathlineFile)

    m_PathlineDataSet = vtk.vtkPolyData()
    m_PathlinePoints  = vtk.vtkPoints()
    m_PathlineLines   = vtk.vtkCellArray()
    m_PathlineScalars = vtk.vtkDoubleArray()
    m_PathlineScalars.SetNumberOfComponents(1)
    m_PathlineDataSet.SetPoints(m_PathlinePoints)
    m_PathlineDataSet.SetLines(m_PathlineLines)
    m_PathlineDataSet.GetPointData().SetScalars(m_PathlineScalars)

    np = m_NumberOfPathlines
    nc = m_NumberOfPathlineCoordinates
    if (np > 0):
        doubleArray = vtk.vtkDoubleArray()
        doubleArray.SetNumberOfComponents(3)
        doubleArray.SetArray(m_PathlineCoordinates, 3 * nc, 1)
        m_PathlinePoints.SetData(doubleArray)
    
        m_PathlineScalars.SetArray(m_PathlineScalarArray, nc, 1)

        intArray = vtk.vtkIdTypeArray()
        intArray.SetNumberOfTuples(1)
        intArray.SetArray(m_PathlinePointArray, np + nc, 1)
        m_PathlineLines.SetCells(np, intArray)

        assert m_PathlineDataSet.GetLines().GetNumberOfCells() == m_NumberOfPathlines
        assert m_PathlineDataSet.GetNumberOfLines() == m_NumberOfPathlines
        assert m_PathlineDataSet.GetPoints().GetNumberOfPoints() == m_NumberOfPathlineCoordinates

    # Can't just return the m_PathlineDataSet
    # seems that m_PathlineCoordinates and m_PathlineScalarArray need to stay in scope
    return (m_PathlineDataSet, m_PathlineCoordinates, m_PathlineScalarArray)

def renderPathlines(pathlineFile, asTubes):

    pathlines = mvPathlines(pathlineFile, asTubes)

    # model viewer mvManager method to scale tube diameter
    bounds = pathlines.GetActor().GetBounds()
    defaultAxesSize = (bounds[1]-bounds[0] + bounds[3]-bounds[2] + bounds[5]-bounds[4])/12
    pathlines.SetDefaultTubeDiameter(defaultAxesSize * 0.1)

    colors = vtk.vtkNamedColors()

    renderer = vtk.vtkRenderer()
    renderWindow = vtk.vtkRenderWindow()
    renderWindow.SetSize(1600, 1080)
    renderWindow.AddRenderer(renderer)

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)

    renderer.AddActor(pathlines.GetActor())

    renderer.SetBackground(colors.GetColor3d("SlateGray"))

    renderWindow.Render()
    renderWindowInteractor.Start()

def main():
    renderPathlines("tc2hufv4.path", False)
    renderPathlines("tc2hufv4.path", True)

if __name__ == '__main__':
    main()
