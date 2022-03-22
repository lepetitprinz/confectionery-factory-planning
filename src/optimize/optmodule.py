import re
import networkx as nx
import datetime as dt


def dps(BOM, item, actlist=None, actname=None):
    if not actlist:
        actlist = []
    if not actname:
        actname = item
        actlist.append(actname)
    else:
        if item in actname:
            actname = actname[actname.index(item) + len(item) + 1:]
        if not actname:
            actname = item
        else:
            actname = item + "_" + actname
        actlist.append(actname)
    L = BOM.predecessors(item)
    l = []
    for i in L:
        l.append((i, len(BOM[i])))
    for n, d in sorted(l, key=lambda x: x[1]):
        dps(BOM, n, actlist, actname)
    return actlist


def make_activities(BOM, item, repeat=1, actlist=None, actname=None, temporal=None):
    """
    BOM: BOMGraph
    item: product
    repeat: BOM.rate
    """
    if not actlist:
        actlist = []
        temporal = []
    for i in range(repeat):
        if not actname:
            actname = item + str(i)
            actlist.append(actname)
        else:
            if item + str(i - 1) in actname:
                actname = actname[actname.index(item + str(i - 1)) + len(item + str(i - 1)) + 1:]
            if not actname:
                actname = item + str(i)
            else:
                temporal.append((item + str(i) + "_" + actname, actname))
                actname = item + str(i) + "_" + actname
            actlist.append(actname)
        for c in BOM.predecessors(item):
            make_activities(BOM, c, BOM[c][item]["rate"], actlist, actname, temporal)
    return actlist, temporal


def base(n, b):
    l = []
    s = 'abcdefghijklmnopqrstuvwxyz'.upper()
    while True:
        l.append(s[n % b])
        if n // b < b:
            l.append(s[n // b])
            break
        n = n // b
    return ''.join(l[::-1])


def make_bom(bomdf, item_code):
    BOM1, BOM2 = nx.DiGraph(), nx.DiGraph()

    for i in bomdf.index:
        BOM1.add_edge(bomdf.CHILD_ITEM[i], bomdf.PARENT_ITEM[i], rate=bomdf.RATE[i])
        BOM2.add_edge(item_code[bomdf.CHILD_ITEM[i]], item_code[bomdf.PARENT_ITEM[i]], rate=bomdf.RATE[i])
    return BOM1, BOM2


def make_reslist(df):
    reslist = []
    df = df.sort_values("OPERATION_NO")
    start = 0
    for i in df.index:
        reslist.append((df.WC_CD[i], int(start), int(start + df.SCHD_TIME[i])))
        start += df.SCHD_TIME[i]
    return reslist


def bestsol_write_csv(L, opdf, code_item):
    f = open("bestsolution.csv", "w")
    f.write("Act,Code,Num,Product,Resource,Start,End,StartTime,EndTime\n")
    now = dt.datetime.now()
    for i in L:
        actname = i[0]
        actno = re.search(r"[0-9]+$", i[0]).group()
        item = re.match(r"[A-Z]+", actname).group()
        times = i[2]
        acts = [tuple(i.split("--")) for i in re.findall(r"[0-9]+\-\-[0-9]+", times)]
        sub_df = opdf[opdf.ITEM_CD == code_item[item]]
        op_no = list(sub_df.OPERATION_NO)
        for s, e in acts:
            s, e = int(s), int(e)
            time = e - s
            while time:
                no = op_no.pop(0)
                now_df = sub_df[sub_df.OPERATION_NO == no]
                sc_time = int(now_df.SCHD_TIME)
                wc_cd = now_df.WC_CD.reset_index(drop=True)[0]
                e = s + sc_time
                stime = now + dt.timedelta(minutes=s)
                etime = now + dt.timedelta(minutes=e)
                f.write(
                    "{},{},{},{},{},{},{},{},{}\n".format(actname + "_" + str(no), item, actno, code_item[item], wc_cd,
                                                          s, e, stime, etime))
                s += sc_time
                time -= sc_time
    f.close()
