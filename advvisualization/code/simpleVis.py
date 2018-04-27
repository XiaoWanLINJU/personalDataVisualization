import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import mpld3
from mpld3 import plugins, utils

from matplotlib.ticker import NullFormatter  # useful for `logit` scale

class personVis:
    def __init__(self):
        self.activityf = "../data/baseline/activities.csv"
        self.sleepf = "../data/baseline/sleep.csv"
        self.deskf =  "../data/baseline/deskclean.csvstatis.csv"
        self.loadAlldata()

    def plotActivity(self, file):
        '''
        for each day, get the time of: Minutes Lightly Active,
        Minutes Fairly Active,Minutes Very Active,Activity Calories

        :param file:(row number is for date),Calories Burned,Steps,Distance,
        Floors,Minutes Sedentary,Minutes Lightly Active,
        Minutes Fairly,Active,Minutes Very Active,Activity Calories

        :return: dic, with key are
        '''
        activity = loadData(file)
        # lines = open(file, 'r').readlines()
        # for i in range(1, len(lines)):
        #     tokes = lines[i].strip().split(',')
        #     activity[i] = [float(t)/60 for t in tokes]
        #     # print(activity[i])
        # print( activity.values(),list(activity.keys()))
        # for k in list(activity.keys()):
        #     plt.hist( activity[k],list(activity.keys()), stacked=True)
        # plt.show()
        c = []

        v0 = []
        v1 = []
        v2 = []
        for key, val in activity.items():
            c.append(key)
            v0.append(val[0])
            v1.append(val[1])
            v2.append(val[2])
        v0 = np.array(v0)
        v1 = np.array(v1)
        v2 = np.array(v2)
        # print(v0)
        # print(v1)
        # print(v2)
        p0 = plt.bar(range(len(c)), v0, linewidth=1.5)
        p1 = plt.bar(range(len(c)), v1, bottom=v0, linewidth=1.5)
        p2 = plt.bar(range(len(c)), v2, bottom=v1 + v0, linewidth=1.5)
        plt.xticks(range(len(c)), c)
        plt.legend((p0[0], p1[0], p2[0]), ('Lightly Active', 'Fairly Active','Very Active'))
        plt.xlabel("days")
        plt.ylabel("hours")
        plt.title("level of activity")
        plt.xticks(np.arange(0, 30, 2), np.arange(0, 30, 2))
        # plt.title("sleep pattern for days in a month")
        plt.xlim(-0.5, 40)
        plt.show()
        return activity

    def statisDesk(self, file):
        lines = open(file, 'r').readlines()
        activity = {}
        for line in lines:
            tokens = line.strip().split(',')
            if tokens[0] not in activity.keys():
                tmp = {'-2':0, '-1':0, '0':0, '1':0, '2':0}
                tmp[tokens[2]] = float(tokens[1])/(60*60)
                activity[tokens[0]] = tmp
            else:
                tmp = activity[tokens[0]]
                tmp[tokens[2]] += float(tokens[1])/(60*60)
                activity[tokens[0]] = tmp
        fw = open(file + "statis.csv", 'w')
        for key in activity.keys():
            times =  activity[key].values()
            fw.write(','.join([str(x) for x in times]) + '\n')
        fw.close()

    def plotSleep(self, file):
        '''
        for each day, get the time of: Minutes Awake,Minutes REM Sleep,
        Minutes Light Sleep, Minutes Deep Sleep

        :param file:(row number is for date),Calories Burned,Steps,Distance,
        Floors,Minutes Sedentary,Minutes Lightly Active,
        Minutes Fairly,Active,Minutes Very Active,Activity Calories

        :return: dic, with key are
        '''
        activity = loadData(file)
        # for k in list(activity.keys()):
        #     plt.hist( activity[k],list(activity.keys()), stacked=True)
        # plt.show()
        c = []
        v0 = []
        v1 = []
        v2 = []
        v3 = []
        for key, val in activity.items():
            c.append(key)
            v0.append(val[0])
            v1.append(val[1])
            v2.append(val[2])
            v3.append(val[3])
        v0 = np.array(v0)
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)
        # print(v0)
        # print(v1)
        # print(v2)
        p3 = plt.bar(range(len(c)), v3, linewidth=3)
        p2 = plt.bar(range(len(c)), v2, bottom=v3, linewidth=3)
        p1 = plt.bar(range(len(c)), v1, bottom=v3 + v2, linewidth=3)
        p0 = plt.bar(range(len(c)), v0, bottom=v2 + v1 + v3, linewidth=3)
        plt.xticks(range(len(c)), c)
        plt.legend((p0[0], p1[0], p2[0], p3[0]), ('awake', 'REM','light sleep', 'deep sleep'))
        plt.xlabel("days")
        plt.ylabel("hours")
        plt.title("sleep pattern for days in a month")
        plt.xticks(np.arange(0, 30, 2), np.arange(0, 30, 2))
        # plt.title("sleep pattern for days in a month")
        plt.xlim(-0.5, 40)
        plt.show()
        return activity

    def plotPro(self):
        '''
        for each day, get the time of: Minutes Awake,Minutes REM Sleep,
        Minutes Light Sleep, Minutes Deep Sleep

        :param file:(row number is for date),Calories Burned,Steps,Distance,
        Floors,Minutes Sedentary,Minutes Lightly Active,
        Minutes Fairly,Active,Minutes Very Active,Activity Calories

        :return: dic, with key are
        '''
        if len(self.desk) < 1:
            self.loadAlldata()
        print(self.desk.values())
        activity = self.desk
        # for k in list(activity.keys()):
        #     plt.hist( activity[k],list(activity.keys()), stacked=True)
        # plt.show()
        c = []
        v0 = []
        v1 = []
        v2 = []
        v3 = []
        v4 = []
        for key, val in activity.items():
            c.append(key)
            v0.append(val[0])
            v1.append(val[1])
            v2.append(val[2])
            v3.append(val[3])
            v4.append(val[4])
        v0 = np.array(v0)
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)
        v4 = np.array(v4)
        # print(v0)
        # print(v1)
        # print(v2)
        p0 = plt.bar(range(len(c)), v4, linewidth=3)
        p1 = plt.bar(range(len(c)), v3, bottom=v4, linewidth=3)
        p2 = plt.bar(range(len(c)), v2, bottom=v4 + v3, linewidth=3)
        p3 = plt.bar(range(len(c)), v1, bottom=v4 + v3 + v2, linewidth=3)
        p4 = plt.bar(range(len(c)), v0, bottom=v4 + v1 + v2 + v3, linewidth=3)
        plt.xticks(range(len(c)), c)
        plt.legend((p4[0], p3[0], p2[0], p1[0], p0[0]), ('very not productive', 'not productive',
                                                         'neutural', 'productive', 'very productive'), loc=1)
        plt.xlabel("days")
        plt.ylabel("hours")
        plt.title("productivity for days in a month")
        plt.xticks(np.arange(0, 30, 2), np.arange(0, 30, 2))
        # plt.title("sleep pattern for days in a month")
        plt.xlim(-0.5, 40)
        plt.ylim(0,20)
        # plt.show()
        # return activity
        return plt

    def loadData(self, file):
        activity = {}
        # print(file)
        lines = open(file, 'r').readlines()
        for i in range(1, len(lines)):
            tokes = lines[i].strip().split(',')
            activity[i] = [float(t) for t in tokes]
            # print(activity[i])
        # print(activity.values(), list(activity.keys()))
        print(file, len(activity[1]))
        return activity




    def exploreSleepActivity(self):
        '''
        assume each day have both data
        :param sleepf:
        :param activityf:
        :return:
        '''

        fig, ax = plt.subplots(3, 3, figsize=(8, 8))
        fig.suptitle("explore the sleep pattern and the activity level", fontsize=16)
        # fig.subplots_adjust(top=0.5, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
        #                     wspace=0.25)

        ax = ax[::-1]
        sleep = self.sleep  # awake, REM, light, deep
        acti = self.activity  # -2, -1,0,1,2
        allDic = {}
        for key in list(sleep.keys()):
            allDic[key] = [x for x in acti[key]] + sleep[key]

        # print(allDic.values())
        lightDic = dict(sorted(allDic.items(), key=lambda d: d[1][0]))

        alldata = np.array(list(lightDic.values()))
        xlabel = ["light activity duration", "fair activity duration", "very activity duration"]
        ylabel = ["REM duration", "light sleep duration", "deep sleep duration"]

        # X = np.random.normal(size=(3, 100))
        for i in range(3):
            for j in range(3):
                ax[i, j].xaxis.set_major_formatter(plt.NullFormatter())
                ax[i, j].yaxis.set_major_formatter(plt.NullFormatter())
                ax[i, j].set(xlabel=xlabel[i], ylabel=ylabel[j])


                points = ax[i, j].scatter(alldata[:, i], alldata[:, 3+j])

        fig.tight_layout()

        plugins.connect(fig, plugins.LinkedBrush(points))
        # mpld3.show()
        mpld3.save_html(fig, 'sleepvsactivity.html')


        # print(alldata[:, 0])
        # plt.plot(alldata)
        # plt.subplot(331)
        # plt.scatter(alldata[:, 0], alldata[:, 4],)
        # plt.ylabel("REM")
        # plt.xlabel("light activity")
        #
        # plt.subplot(332)
        # plt.scatter(alldata[:, 0], alldata[:, 5], )
        # plt.ylabel("light sleep")
        # plt.xlabel("light activity")
        #
        # plt.subplot(333)
        # plt.scatter(alldata[:, 0], alldata[:, 6], )
        # plt.ylabel("deep sleep")
        # plt.xlabel("light activity")
        #
        # lightDic = dict(sorted(allDic.items(), key=lambda d: d[1][1]))
        #
        # alldata = np.array(list(lightDic.values()))
        # # print(alldata[:, 0])
        # # plt.plot(alldata)
        # plt.subplot(334)
        # plt.scatter(alldata[:, 1], alldata[:, 4], )
        # plt.ylabel("REM")
        # plt.xlabel("fair activity")
        #
        # plt.subplot(335)
        # plt.scatter(alldata[:, 1], alldata[:, 5], )
        # plt.ylabel("light sleep")
        # plt.xlabel("fair activity")
        #
        # plt.subplot(336)
        # plt.scatter(alldata[:, 1], alldata[:, 6], )
        # plt.ylabel("deep sleep")
        # plt.xlabel("fair activity")
        #
        # lightDic = dict(sorted(allDic.items(), key=lambda d: d[1][2]))
        #
        # alldata = np.array(list(lightDic.values()))
        # # print(alldata[:, 0])
        # # plt.plot(alldata)
        # plt.subplot(337)
        # plt.scatter(alldata[:, 2], alldata[:, 3], )
        # plt.ylabel("REM")
        # plt.xlabel()
        #
        # plt.subplot(338)
        # plt.scatter(alldata[:, 2], alldata[:, 4], )
        # plt.ylabel("light sleep")
        # plt.xlabel("very activity")
        #
        # plt.subplot(339)
        # plt.scatter(alldata[:, 2], alldata[:, 5], )
        # plt.ylabel("deep sleep")
        # plt.xlabel("very activity")
        #
        # # Format the minor tick labels of the y-axis into empty strings with
        # # `NullFormatter`, to avoid cumbering the axis with too many labels.
        # plt.gca().yaxis.set_minor_formatter(NullFormatter())
        # # Adjust the subplot layout, because the logit one may take more space
        # # than usual, due to y-tick labels like "1 - 10^{-3}"
        # plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
        #                     wspace=0.25)
        #
        # plt.show()

    def exploreSleepPro(self):
        '''
        assume each day have both data
        :param sleepf:
        :param activityf:
        :return:
        '''
        sleep = self.sleep# awake, REM, light, deep
        pro = self.desk# light, fair, very
        allDic = {}
        for key in list(sleep.keys()):
            allDic[key] = [x for x in pro[key]] + [sum(x for x in pro[key])] + sleep[key]
        # print(allDic.values())
        lightDic = dict(sorted(allDic.items(), key=lambda d:d[1][0]))

        alldata = np.array(list(lightDic.values()))
        fig, ax = plt.subplots(3, 3, figsize=(8, 8))

        ax = ax[::-1]

        xlabel = ["not productive duration", "neutural duration", "productive duration"]
        ylabel = ["REM duration", "light sleep duration", "deep sleep duration"]

        # X = np.random.normal(size=(3, 100))
        for i in range(3):
            for j in range(3):
                ax[i, j].xaxis.set_major_formatter(plt.NullFormatter())
                # ax[i, j].yaxis.set_major_formatter(plt.NullFormatter())
                ax[i, j].set(xlabel=xlabel[i], ylabel=ylabel[j])
                if i != 1:
                    points = ax[i, j].scatter(alldata[:, i*2] + alldata[:, i*2 + 1], alldata[:, 7 + j],)
                else:
                    points = ax[i, j].scatter(alldata[:, i * 2], alldata[:, 7 + j], )
        # points = ax[0, 0].scatter(alldata[:, 0] + alldata[:, 1], alldata[:, 7],)
        # points = ax[0, 1].scatter(alldata[:, 0] + alldata[:, 1], alldata[:, 8], )
        # points = ax[0, 2].scatter(alldata[:, 0] + alldata[:, 1], alldata[:, 9], )
        # lightDic = dict(sorted(allDic.items(), key=lambda d: d[1][2]))
        #
        # alldata = np.array(list(lightDic.values()))
        # points = ax[1, 0].scatter(alldata[:, 2], alldata[:, 7], )
        # points = ax[1, 1].scatter(alldata[:, 2], alldata[:, 8], )
        # points = ax[1, 2].scatter(alldata[:, 2], alldata[:, 9], )
        # lightDic = dict(sorted(allDic.items(), key=lambda d: d[1][4]))
        # alldata = np.array(list(lightDic.values()))
        # points = ax[2, 0].scatter(alldata[:, 4] + alldata[:, 5], alldata[:, 7], )
        # points = ax[2, 1].scatter(alldata[:, 4] + alldata[:, 5], alldata[:, 8], )
        # points = ax[2, 2].scatter(alldata[:, 4] + alldata[:, 5], alldata[:, 9], )

        plugins.connect(fig, plugins.LinkedBrush(points))
        # mpld3.show()
        mpld3.save_html(fig, 'sleepvspro.html')

    def loadAlldata(self):
        self.activity = self.loadData(self.activityf)
        self.sleep = self.loadData(self.sleepf)
        self.desk = self.loadData(self.deskf)
        self.daylog = {}
        self.dayempty = {}
        # print(self.activity)
        for i in self.activity.keys():
            self.daylog[i] =[sum(list(self.activity[i]))/60, sum(list(self.sleep[i]))/60, sum(list(self.desk[i])),
                             sum(list(self.activity[i]))/60 + sum(list(self.sleep[i]))/60 + sum(list(self.desk[i]))]
            self.dayempty[i] = 24 - self.daylog[i][-1]
        # print(self.day)

    # def wholeday(self):
    #     self.loadAlldata()
    #
    #     fig, ax = plt.subplots(2)
    #     c = np.array([int(x) for x in self.daylog.keys()])
    #     print(c)
    #     P = range(len(c))
    #     A = np.array(list(self.daylog.values()))[:,-1]
    #     # print(np.array(list(self.daylog.values())))
    #     # print(np.array(list(self.daylog.values()))[:,-1])
    #     p3 = ax[0].scatter(P, A, c=A,
    #                        s=200, alpha=0.5)
    #     # scatter(, , linewidth=3)
    #     # p2 = ax[0].bar(range(len(c)), np.array(list(self.dayempty.values())), bottom=np.array(list(self.daylog.values()))[:,-1], linewidth=3)
    #     # plt.xticks(range(len(c)), c)
    #     # plt.legend((p3[0], p2[0]), ('loged', 'not logged'), loc=1)
    #     # plt.xlabel("days")
    #     # plt.ylabel("hours")
    #     # plt.title("logged time for days in a month")
    #     # plt.xticks(np.arange(0, 30, 2), np.arange(0, 30, 2))
    #     # # plt.title("sleep pattern for days in a month")
    #     # plt.xlim(-0.5, 31)
    #     # plt.ylim(0, 40)
    #
    #     # plt.subplot(212)
    #     # print(type(c))
    #     lines = ax[1].plot(c[:3], 0 * c[:3], '-w', lw=3, alpha=0.5)
    #     # ax[1].set_ylim(-1, 1)
    #
    #     ax[1].set_title("Hover over points to see lines")
    #
    #     # labels = 'workout', 'sleep', 'login desktop'
    #     # sizes = [0,0,0]
    #     # colors = ['yellowgreen', 'lightcoral', 'lightskyblue']
    #     # explode = ( 0, 0, 0)  # explode 1st slice
    #     #
    #     # # Plot
    #     # pies = ax[1].pie(sizes, explode=explode, labels=labels, colors=colors,
    #     #         autopct='%1.1f%%', shadow=True, startangle=140)
    #
    #     # # scatter periods and amplitudes
    #     # lines = ax[0].plot(x, 0 * x, '-w', lw=3, alpha=0.5)
    #     # ax[0].set_ylim(-1, 1)
    #     #
    #     # ax[0].set_title("Hover over points to see lines")
    #     #
    #     # # transpose line data and add plugin
    #
    #     P = 0.2 + np.random.random(size=30)
    #     A = np.random.random(size=30)
    #     x = np.linspace(0, 10, 100)
    #     print(type(x), type(c))
    #
    #     data = np.array([[x, Ai * np.sin(x / Pi)]
    #                      for (Ai, Pi) in zip(A, P)])
    #     print(data.shape)
    #
    #     # A2 = np.array(list(self.daylog.values()))[:,0: 3]
    #     # print(A2)
    #     # piedata = np.array([[c, Ai] for Ai in A2])
    #     # print(piedata.shape)
    #
    #     linedata = data.transpose(0, 2, 1)
    #     # print(type(p3), type(pies[0]))
    #     print(linedata.shape)
    #     plugins.connect(fig, LinkedView(p3, lines[0], linedata))
    #
    #     mpld3.show()
    def wholeday(self):
        self.loadAlldata()

        # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig = plt.figure(figsize=(8, 24))  # 8 inches wide by 6 inches tall
        ax1 = fig.add_subplot(4, 1, 1)
        ax2 = fig.add_subplot(4, 1, 2)
        ax3 = fig.add_subplot(4, 1, 3)
        ax4 = fig.add_subplot(4, 1, 4)
        c = np.array([int(x) for x in self.daylog.keys()])
        # print(c)
        P = range(len(c))
        A = np.array(list(self.daylog.values()))[:,-1]
        # fig, ax = plt.subplots()
        points = ax1.scatter(P, A,c=A,
                           s=100, alpha=0.5)
        labels = ["login time: {0}".format(i) for i in A]
        # print(labels)
        ax1.set_title("whole log in time")
        # ax1.set_xlabel("days")
        ax1.set_ylabel("hours")

        if len(self.desk) < 1:
            self.loadAlldata()
        print(self.desk.values())
        activity = self.desk
        # for k in list(activity.keys()):
        #     plt.hist( activity[k],list(activity.keys()), stacked=True)
        # plt.show()
        c = []
        v0 = []
        v1 = []
        v2 = []
        v3 = []
        v4 = []
        for key, val in activity.items():
            c.append(key)
            v0.append(val[0])
            v1.append(val[1])
            v2.append(val[2])
            v3.append(val[3])
            v4.append(val[4])
        v0 = np.array(v0)
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)
        v4 = np.array(v4)
        # print(v0)
        # print(v1)
        # print(v2)
        p0 = ax2.bar(range(len(c)), v4, linewidth=3)
        p1 = ax2.bar(range(len(c)), v3, bottom=v4, linewidth=3)
        p2 = ax2.bar(range(len(c)), v2, bottom=v4 + v3, linewidth=3)
        p3 = ax2.bar(range(len(c)), v1, bottom=v4 + v3 + v2, linewidth=3)
        p4 = ax2.bar(range(len(c)), v0, bottom=v4 + v1 + v2 + v3, linewidth=3)
        # ax2.xticks(range(len(c)), c)
        ax2.legend((p4[0], p3[0], p2[0], p1[0], p0[0]), ('very not productive', 'not productive',
                                                         'neutural', 'productive', 'very productive'), loc=1)
        # ax2.set_xlabel("days")
        ax2.set_ylabel("hours")
        ax2.set_title("productivity for days in a month")
        # ax2.set_xticks(np.arange(0, 30, 2), np.arange(0, 30, 2))
        # plt.title("sleep pattern for days in a month")
        ax2.set_xlim(-0.5, 30)
        ax2.set_ylim(0,24)



        tooltip = plugins.PointLabelTooltip(points, labels)

        plugins.connect(fig, tooltip)

        activity = self.activity
        # lines = open(file, 'r').readlines()
        # for i in range(1, len(lines)):
        #     tokes = lines[i].strip().split(',')
        #     activity[i] = [float(t)/60 for t in tokes]
        #     # print(activity[i])
        # print( activity.values(),list(activity.keys()))
        # for k in list(activity.keys()):
        #     plt.hist( activity[k],list(activity.keys()), stacked=True)
        # plt.show()
        c = []

        v0 = []
        v1 = []
        v2 = []
        for key, val in activity.items():
            c.append(key)
            v0.append(val[0]/60)
            v1.append(val[1]/60)
            v2.append(val[2]/60)
        v0 = np.array(v0)
        v1 = np.array(v1)
        v2 = np.array(v2)
        # print(v0)
        # print(v1)
        # print(v2)
        p0 = ax3.bar(range(len(c)), v0, linewidth=1.5)
        p1 = ax3.bar(range(len(c)), v1, bottom=v0, linewidth=1.5)
        p2 = ax3.bar(range(len(c)), v2, bottom=v1 + v0, linewidth=1.5)
        ax3.set_xticks(range(len(c)), c)
        ax3.legend((p0[0], p1[0], p2[0]), ('Lightly Active', 'Fairly Active', 'Very Active'))
        ax3.set_xlabel("days")
        ax3.set_ylabel("hours")
        ax3.set_title("level of activity")
        # ax3.set_xticks(np.arange(0, 30, 2), np.arange(0, 30, 2))
        # plt.title("sleep pattern for days in a month")
        ax3.set_xlim(-0.5, 30)
        ax3.set_ylim(0, 24)

        activity = self.sleep
        # for k in list(activity.keys()):
        #     plt.hist( activity[k],list(activity.keys()), stacked=True)
        # plt.show()
        c = []
        v0 = []
        v1 = []
        v2 = []
        v3 = []
        for key, val in activity.items():
            c.append(key)
            v0.append(val[0]/60)
            v1.append(val[1]/60)
            v2.append(val[2]/60)
            v3.append(val[3]/60)
        v0 = np.array(v0)
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)
        # print(v0)
        # print(v1)
        # print(v2)
        p3 = ax4.bar(range(len(c)), v3, linewidth=3)
        p2 = ax4.bar(range(len(c)), v2, bottom=v3, linewidth=3)
        p1 = ax4.bar(range(len(c)), v1, bottom=v3 + v2, linewidth=3)
        p0 = ax4.bar(range(len(c)), v0, bottom=v2 + v1 + v3, linewidth=3)
        # ax4.xticks(range(len(c)), c)
        ax4.legend((p0[0], p1[0], p2[0], p3[0]), ('awake', 'REM', 'light sleep', 'deep sleep'))
        ax4.set_xlabel("days")
        ax4.set_ylabel("hours")
        ax4.set_title("sleep pattern for days in a month")
        # ax4.xticks(np.arange(0, 30, 2), np.arange(0, 30, 2))
        # plt.title("sleep pattern for days in a month")
        ax4.set_xlim(-0.5, 30)
        ax4.set_ylim(0, 24)

        # mpld3.show()
        mpld3.save_html(fig, "wholeday.html")

    def allTrend(self):
        self.loadAlldata()
        fig = plt.figure(figsize=(17, 13))  # 8 inches wide by 6 inches tall
        ax = fig.add_subplot(2, 2, 1)
        ax.set_ylabel('hours')
        ax.set_xlabel('days in a month')
        ax.set_ylim(0, 12)
        ax.set_title('time trend for different activity')
        labels = [ "light activity", "fairly activity", "very activity",
                   'very not productive', 'not productive', 'neutural', 'productive', 'very productive',
                   'awake', 'REM', 'light sleep', 'deep sleep']

        colors = [ 'lime', 'green', 'darkgreen',
                   'gray', 'brown', 'blue', 'darkblue', 'navy',
                   'coral', 'violet', 'plum', 'purple']
        line_collections = []
        activityvalue = np.array(list(self.activity.values()))
        for i in range(3):
            y1 = activityvalue[:, i]/60
            x1 = np.array(range(len(y1)))# light activity, fairly activity, very activity
            l1 = ax.plot(x1, y1, lw=3, alpha=0.4, c=colors[i], label=labels[i])
            line_collections.append(l1)

        provalues = np.array(list(self.desk.values()))
        for i in range(5):
            y1 = provalues[:, i]
            x1 = np.array(range(len(y1)))  #
            l1 = ax.plot(x1, y1, lw=3, alpha=0.4, c=colors[ 3 + i], label=labels[3 + i])
            line_collections.append(l1)

        sleepvalues = np.array(list(self.sleep.values()))

        for i in range(4):
            y1 = sleepvalues[:, i]/60
            x1 = np.array(range(len(y1)))#
            l1 = ax.plot(x1, y1, lw=4, alpha=0.4, c=colors[8 + i], label=labels[8 + i])
            line_collections.append(l1)

        plugins.connect(fig, plugins.InteractiveLegendPlugin(line_collections, labels))

        # mpld3.show()
        mpld3.save_html(fig, 'trend.html')

    def testActiveLegend(self):
        import mpld3
        from mpld3 import plugins
        from mpld3.utils import get_id
        import numpy as np
        import collections
        import matplotlib.pyplot as plt

        N_paths = 5
        N_steps = 100

        x = np.linspace(0, 10, 100)
        y = 0.1 * (np.random.random((N_paths, N_steps)) - 0.5)
        y = y.cumsum(1)
        fig = plt.figure(figsize=(16, 7))  # 8 inches wide by 6 inches tall
        ax = fig.add_subplot(2, 2, 1)

        # fig, ax = plt.subplots()
        labels = ["a", "b", "c", "d", "e"]
        line_collections = ax.plot(x, y.T, lw=4, alpha=0.2)
        interactive_legend = plugins.InteractiveLegendPlugin(line_collections, labels)
        plugins.connect(fig, interactive_legend)

        mpld3.show()

    def testBrush(self):
        fig, ax = plt.subplots(3, 3, figsize=(6, 6))
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        ax = ax[::-1]

        X = np.random.normal(size=(3, 100))
        for i in range(3):
            for j in range(3):
                ax[i, j].xaxis.set_major_formatter(plt.NullFormatter())
                ax[i, j].yaxis.set_major_formatter(plt.NullFormatter())
                points = ax[i, j].scatter(X[j], X[i])

        plugins.connect(fig, plugins.LinkedBrush(points))
        mpld3.show()

class LinkedViewpie(plugins.PluginBase):
    """A simple plugin showing how multiple axes can be linked"""

    JAVASCRIPT = """
    mpld3.register_plugin("linkedview", LinkedViewPlugin);
    LinkedViewPlugin.prototype = Object.create(mpld3.Plugin.prototype);
    LinkedViewPlugin.prototype.constructor = LinkedViewPlugin;
    LinkedViewPlugin.prototype.requiredProps = ["idpts", "idbar","bardata"];
    LinkedViewPlugin.prototype.defaultProps = {}
    function LinkedViewPlugin(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };
    
    LinkedViewPlugin.prototype.draw = function(){
    console.log(this.props.idbar)
      var pts = mpld3.get_element(this.props.idpts);
      var bar = mpld3.get_element(this.props.idbar);
      var data = this.props.data;
    console.log(data)
      function mousedown(d, i){
      
      console.log(pts)
      console.log(bar)
        bar.data = data[i];
        bar.elementsl().transition()
            .attr("d", bar.datafunc(bar.data))
            .style("stroke", this.style.fill);
      }
     
      pts.elements().on("mousedown", mousedown);
    };
    """

    def __init__(self, points, bar, bardata):
        if isinstance(points, matplotlib.lines.Line2D):
            suffix = "pts"
        else:
            suffix = None

        self.dict_ = {"type": "linkedview",
                      "idpts": utils.get_id(points, suffix),
                      "idbar": utils.get_id(bar),
                      "data": bardata}


def test():
    fig, ax = plt.subplots(2)

    # scatter periods and amplitudes
    np.random.seed(0)
    P = 0.2 + np.random.random(size=20)
    A = np.random.random(size=20)
    x = np.linspace(0, 10, 100)
    data = np.array([[x, Ai * np.sin(x / Pi)]
                     for (Ai, Pi) in zip(A, P)])
    points = ax[1].scatter(P, A, c=P + A,
                           s=200, alpha=0.5)
    ax[1].set_xlabel('Period')
    ax[1].set_ylabel('Amplitude')

    # create the line object
    # print(type(x))
    lines = ax[0].plot(x, 0 * x, '-w', lw=3, alpha=0.5)
    ax[0].set_ylim(-1, 1)

    ax[0].set_title("Hover over points to see lines")

    # transpose line data and add plugin
    # print(data.shape)
    linedata = data.transpose(0, 2, 1)
    print(linedata.shape)
    # linedata = linedata.tolist()
    # print(len(linedata[0]))
    print(type(lines[0]), lines[0])
    plugins.connect(fig, LinkedView(points, lines[0], linedata))

    mpld3.show()


def testInterLegend():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import mpld3
    from mpld3 import plugins
    np.random.seed(9615)

    # generate df
    N = 100
    df = pd.DataFrame((.1 * (np.random.random((N, 5)) - .5)).cumsum(0),
                      columns=['a', 'b', 'c', 'd', 'e'], )

    # plot line + confidence interval
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)

    for key, val in df.iteritems():
        l, = ax.plot(val.index, val.values, label=key)
        ax.fill_between(val.index,
                        val.values * .5, val.values * 1.5,
                        color=l.get_color(), alpha=.4)

    # define interactive legend

    handles, labels = ax.get_legend_handles_labels()  # return lines and labels
    interactive_legend = plugins.InteractiveLegendPlugin(zip(handles,
                                                             ax.collections),
                                                         labels,
                                                         alpha_unsel=0.5,
                                                         alpha_over=1.5,
                                                         start_visible=True)
    plugins.connect(fig, interactive_legend)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Interactive legend', size=20)

    mpld3.show()

if __name__=="__main__":
    # plotActivity("../data/baseline/activities.csv")
    # plotSleep("../data/baseline/sleep.csv")
    #
    # testInterLegend()
    # exploreSleepActivity("../data/baseline/sleep.csv", "../data/baseline/activities.csv")
    # statisDesk("../data/baseline/deskclean.csv")
    # plotPro("../data/baseline/deskclean.csvstatis.csv")
    # exploreSleepPro("../data/baseline/sleep.csv", "../data/baseline/deskclean.csvstatis.csv")
    pv = personVis()
    pv.wholeday()
    # pv.allTrend()
    # pv.testActiveLegend()
    # pv.testBrush()
    # pv.exploreSleepActivity()
    # pv.exploreSleepPro()