# Package importation
import numpy as np
import cv2
import time
from operator import itemgetter
# from openpyxl import Workbook
import simpleaudio as sa
def play_audio(c):
        filename = c
        wave_obj = sa.WaveObject.from_wave_file(filename)
        play_obj = wave_obj.play()
        play_obj.wait_done()  # Wait until sound has finished playing

def auditory_navi(deg_div,dista):
            dist={
            "1":"E:/Pankaj_EDI/1m.wav",
            "2":"E:/Pankaj_EDI/2m.wav",
            "3":"E:/Pankaj_EDI/3m.wav",
            "4":"E:/Pankaj_EDI/4m.wav",
            "5":"E:/Pankaj_EDI/5m.wav",
            "6":"E:/Pankaj_EDI/6m.wav",
            "7":"E:/Pankaj_EDI/7m.wav",
            "8":"E:/Pankaj_EDI/8m.wav",
            }

            degree_dev={
            "50":"E:/Pankaj_EDI/SlightlyRight.wav",
            "55":"E:/Pankaj_EDI/SlightlyLeft.wav",
            "60":"E:/Pankaj_EDI/right.wav",
            "70":"E:/Pankaj_EDI/left.wav",
            "80":"E:/Pankaj_EDI/turn.wav",
            }
            d=0;
            degrees=float(deg_div)

            if abs(degrees) <=0.1:
               d=5
            elif degrees >=0.1 and degrees<0.9:
               d=50
            elif degrees <= -0.1 and degrees> -0.9:
               d=55
            elif degrees>0.9:
               d=60
            elif degrees<-0.9:
               d=70


            if d<=5:
               x=int(dista)
               if x==0:
                 exit
               play_audio(dist[str(x)])
            else:
               play_audio("E:/Pankaj_EDI/turn.wav")
               play_audio(degree_dev[str(d)])
            #print('Playsound done')

def funspeech():
    thisdict = {"bottle": 1, "book": 2, "chair": 3, "table": 4, "box": 5, "door": 6, "mobile": 7, "bed": 8, "stick": 9,
                "person": 10}

    import speech_recognition as sr

    r = sr.Recognizer()
    r.pause_threshold = 0.6
    with sr.Microphone() as source:
        audio = r.adjust_for_ambient_noise(source)
        ##########
        play_audio('E:/Pankaj_EDI/listening.wav')######################## 1.
        #print("listening...")
        
        play_audio('E:/warningbeep.wav')
        audio = r.listen(source, timeout=2)

        try:

            print("hold on ... converting...")
            text = r.recognize_google(audio)
            print(text)
            token = thisdict[text]
            print(token)
            return(token)

        except LookupError:
            print("cannot recognise")
            #print("token=100")
            return(100)

def object_detection(token):
    ###Update the dictionary with the required object Templates
    dict = {
    1:'E:/dabba.jpeg',
    2:'E:/template.jpeg',
    3:'E:/rpi_box.jpeg',
    4:'/home/pi/Desktop/images2/',
    5:'/home/pi/Desktop/images2/',
    6:'/home/pi/Desktop/images2/',
    7:'/home/pi/Desktop/images2/',
    8:'/home/pi/Desktop/images2/',
    9:'/home/pi/Desktop/images2/',
    10:'/home/pi/Desktop/images2/',
    }
    path=dict.get(token)
    img = cv2.imread(path,0)
    img = cv2.resize(img,(180,322))
    orb = cv2.ORB_create(nfeatures=500)    #Initializing ORB and no of features
    orb1 = cv2.ORB_create(nfeatures=10000)
    kp_img, des_img = orb.detectAndCompute(img, None)
    #img2 = cv2.drawKeypoints(img,kp_img,None,color=(0,255,0))
    #cv2.destroyAllWindows()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    flagimg = 0
    Nthreshold = 12
    Dthreshold = 40
    check, frame = cap.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_grayframe, des_grayframe = orb1.detectAndCompute(grayframe, None)
    if len(kp_grayframe)!=0:
        matches = bf.match(des_img,des_grayframe)
        nmatches = []
        for m in matches:
            if m.distance<Dthreshold:
                nmatches.append(m)
        dmatches = sorted(nmatches, key = lambda x:x.distance)
        prelim = cv2.drawMatches(img, kp_img, grayframe, kp_grayframe,dmatches[:15],grayframe,flags = 2) #testing
        cv2.imshow("prelim",prelim)
        cv2.waitKey(0)
        print(len(dmatches))
        if len(dmatches)>Nthreshold:
            if flagimg == 1:
                cv2.destroyAllWindows()
            flagimg = 0
            print("found")
            #play_audio('E:/ObjectFound.wav')
            list_kpframe = [kp_grayframe[mat.trainIdx].pt for mat in dmatches]
            X = list(map(itemgetter(0), list_kpframe))
            Y = list(map(itemgetter(1), list_kpframe))
            avg_X = sum(X) / len(X)
            avg_Y = sum(Y) / len(Y)
            #print(avg_X, avg_Y)
            angle_disparity=(avg_X-370)/320
            cap.release()
            #print('Object detection done')
            return avg_X, avg_Y,angle_disparity

        else:
            if flagimg == 0:
                cv2.destroyAllWindows()
            flagimg = 1
            #print("Not found")
            cap.release()
            #print('Object detection done')
            return 0,0,0

kernel = np.ones((3, 3), np.uint8)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

objpoints = []  # 3d points in real world space
imgpointsR = []  # 2d points in image plane
imgpointsL = []
MLS = np.load('E:/EDI_parameters/MLS.npy')
dLS = np.load('E:/EDI_parameters/dLS.npy')
RL = np.load('E:/EDI_parameters/RL.npy')
PL = np.load('E:/EDI_parameters/PL.npy')
RR = np.load('E:/EDI_parameters/RR.npy')
PR = np.load('E:/EDI_parameters/PR.npy')
SIZEL = np.load('E:/EDI_parameters/SIZEL.npy')
SIZER = np.load('E:/EDI_parameters/SIZER.npy')
MRS = np.load('E:/EDI_parameters/MRS.npy')
dRS = np.load('E:/EDI_parameters/dRS.npy')
SIZEL = tuple(SIZEL)  ##These parameters should be tuple
SIZER = tuple(SIZER)
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, SIZEL,cv2.CV_16SC2)  # cv2.CV_16SC2 this format enables us the programme to work faster
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, SIZER, cv2.CV_16SC2)
window_size = 10
min_disp = 2
num_disp = 130 - min_disp
stereO = cv2.StereoSGBM_create(minDisparity=min_disp, numDisparities=num_disp, blockSize=window_size,uniquenessRatio=10, speckleWindowSize=100, speckleRange=32, disp12MaxDiff=5,P1=8 * 3 * window_size ** 2, P2=32 * 3 * window_size ** 2)
stereoR = cv2.ximgproc.createRightMatcher(stereO)  # Create another stereo for right this time
lmbda = 80000
sigma = 1.8
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereO)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
flag = 0
flag_rotation=0
while(True):
    if (flag==0):
        #token=funspeech()
        flag=1
        flag_rotation=0
    if (False):
        ############ 2.
        #play_audio('E:/Pankaj_EDI/NotInAvailableList.wav')
        #print("Play Sound For object Not in available list")
        flag=0
        flag_rotation=0
    else :
        x,y,deg_div=object_detection(1)
        x=int(x)
        y=int(y)
        if ((x!=0) and (y!=0)):
            flag_rotation=0
            CamL = cv2.VideoCapture(0)
            CamR = cv2.VideoCapture(1)                                          # Wenn 0 then Right Cam and wenn 2 Left Cam
            CamL.set(3,640)
            CamL.set(4,480)
            CamR.set(3,640)
            CamR.set(4,480)
            retR, frameR = CamR.read()
            retL, frameL = CamL.read()
            Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT,0)  # Rectify the image using the kalibration parameters founds during the initialisation
            Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1], cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
            Left_nice1 = frameL                                                 # Rectify the image using the kalibration parameters founds during the initialisation
            Right_nice1 = frameR
            grayR = cv2.cvtColor(Right_nice1, cv2.COLOR_BGR2GRAY)
            grayL = cv2.cvtColor(Left_nice1, cv2.COLOR_BGR2GRAY)
            disp = stereO.compute(grayL, grayR)
            #dispL = disp
            #dispR = stereoR.compute(grayR, grayL)
            #dispL = np.int16(dispL)
            #dispR = np.int16(dispR)

            #filteredImg = wls_filter.filter(dispL, grayL, None, dispR)
            #filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
            #filteredImg = np.uint8(filteredImg)
            disp = ((disp.astype(np.float32) / 16) - min_disp) / num_disp
            #closing = cv2.morphologyEx(disp, cv2.MORPH_CLOSE,kernel)            # Apply an morphological filter for closing little "black" holes in the picture(Remove noise)
            #dispc = (closing - closing.min()) * 255
            #dispC = dispc.astype(np.uint8)                                      # Convert the type of the matrix from float32 to uint8, this way you can show the results with the function cv2.imshow()
            #disp_Color = cv2.applyColorMap(dispC, cv2.COLORMAP_JET)             #Change the Color of the Picture into an Ocean Color_Map
            #filt_Color = cv2.applyColorMap(filteredImg, cv2.COLORMAP_JET)
            average = 0
            for u in range(-1, 2):
                for v in range(-1, 2):
                    average += disp[y + u, x + v]
            average = average / 9
            Distance = -593.97 * average ** (3) + 1506.8 * average ** (2) - 1373.1 * average + 522.06
            Distance = np.around(Distance * 0.01, decimals=2)
            print('Distance: '+str(Distance) +' m')
            CamR.release()
            CamL.release()
            print('Disparity done')
            print(deg_div)
            if (Distance>0.7):
                auditory_navi(deg_div,Distance)
            else :
                ################ 3.
                play_audio('E:/Pankaj_EDI/ObjectIsJustAhead.wav')
                #print("Play Sound for object is just ahead")
                flag=0
            time.sleep(int(1*Distance))
        else:
            flag_rotation+=1
            ################### 4.
            play_audio('E:/Pankaj_EDI/TurnRightBy10.wav')
             #print("Play Sound for turn right by 10*")
            if (flag_rotation>=36):
                ################### 5.
                play_audio('E:/Pankaj_EDI/ObjectNotInRoom.wav')
                #print("Play sound for object not in room")
                flag=0
