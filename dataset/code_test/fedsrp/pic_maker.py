import matplotlib.pyplot as plt
import numpy as np

def mk_nse_pic(basin_id, nse_list):
    local_nse = []
    mutil_fune_nse = []
    transfer_a_nse = []
    transfer_b_nse = []
    srp_nse = []
    for item in nse_list:
        local_nse.append(item[0])
        mutil_fune_nse.append(item[1])
        transfer_a_nse.append(item[2])
        transfer_b_nse.append(item[3])
        srp_nse.append(item[4])
    plt.clf()
    width = 1
    plt.plot(local_nse, color=(210/255,180/255,140/255), linewidth=width, marker='o')
    plt.plot(mutil_fune_nse, color=(56/255,232/255,176/255), linewidth=width, marker='*')
    plt.plot(transfer_a_nse, color=(27/255,175/255,208/255), linewidth=width, marker='^')
    plt.plot(transfer_b_nse, color=(105/255,103/255,206/255), linewidth=width, marker='D')
    plt.plot(srp_nse, color=(253/255,99/255,107/255), linewidth=width, marker='s')
    # plt.legend()
    # plt.gca().set_xticklabels([''] * len(plt.gca().get_xticks()))
    # plt.gca().set_yticklabels([''] * len(plt.gca().get_yticks()))
    plt.savefig('/home/FedHydro/dataset/code_test/fedsrp/nse_' + basin_id + '.pdf', dpi=300)
    print('basin:' + basin_id + 'is finished')

def mk_rmse_pic(basin_id, rmse_list, value):
    local_rmse = []
    mutil_fune_rmse = []
    transfer_a_rmse = []
    transfer_b_rmse = []
    srp_rmse = []
    avg_rmse = []
    for item in rmse_list:
        local_rmse.append(item[0])
        mutil_fune_rmse.append(item[1])
        transfer_a_rmse.append(item[2])
        transfer_b_rmse.append(item[3])
        srp_rmse.append(item[4])
        avg_rmse.append(item[5])
    plt.clf()
    width = 1
    plt.plot(local_rmse, color=(210/255,180/255,140/255), linewidth=width, marker='o')
    plt.plot(mutil_fune_rmse, color=(56/255,232/255,176/255), linewidth=width, marker='*')
    plt.plot(transfer_a_rmse, color=(27/255,175/255,208/255), linewidth=width, marker='^')
    plt.plot(transfer_b_rmse, color=(105/255,103/255,206/255), linewidth=width, marker='D')
    plt.plot(avg_rmse, color=(169/255,169/255,169/255), linewidth=width, marker='p')
    plt.plot(srp_rmse, color=(253/255,99/255,107/255), linewidth=width, marker='s')

    num_ticks = 6  # 设置刻度的个数
    plt.xlim(-0.5, 5.5)
    min = 0.002
    max = 0.006
    plt.ylim(min, max)

    y_ticks = np.linspace(min, max, num_ticks)
    plt.yticks(y_ticks)


    plt.grid(linewidth=0.5)

    if not value:
        plt.gca().set_xticklabels([''] * len(plt.gca().get_xticks()))
        plt.gca().set_yticklabels([''] * len(plt.gca().get_yticks()))
    plt.savefig('/home/FedHydro/dataset/code_test/fedsrp/yyy_rmse_' + basin_id + '.jpg', dpi=600)
    print('basin:' + basin_id + 'is finished')

def mk_mae_pic(basin_id, mae_list, value):
    local_mae = []
    mutil_fune_mae = []
    transfer_a_mae = []
    transfer_b_mae = []
    srp_mae = []
    avg_mae = []
    for item in mae_list:
        local_mae.append(item[0])
        mutil_fune_mae.append(item[1])
        transfer_a_mae.append(item[2])
        transfer_b_mae.append(item[3])
        srp_mae.append(item[4])
        avg_mae.append(item[5])
    plt.clf()
    width = 1
    plt.plot(local_mae, color=(210/255,180/255,140/255), linewidth=width, marker='o')
    plt.plot(mutil_fune_mae, color=(56/255,232/255,176/255), linewidth=width, marker='*')
    plt.plot(transfer_a_mae, color=(27/255,175/255,208/255), linewidth=width, marker='^')
    plt.plot(transfer_b_mae, color=(105/255,103/255,206/255), linewidth=width, marker='D')
    plt.plot(avg_mae, color=(169/255,169/255,169/255), linewidth=width, marker='p')
    plt.plot(srp_mae, color=(253/255,99/255,107/255), linewidth=width, marker='s')

    # plt.legend()

    #plt.xlim(-0.5, 5.5)
    #plt.ylim(0.0015, 0.006)
    #plt.gca().yaxis.set_major_locator(plt.MaxNLocator(nbins=6))

    #num_ticks = 6
    #max_num = 0.006
    #min_num = 0.0015
    #step = ((max_num - min_num) / (num_ticks - 1))
    #yticks = np.arange(min_num, max_num + step, step)
    #plt.yticks(yticks)


    num_ticks = 6  # 设置刻度的个数
    plt.xlim(-0.5, 5.5)
    min = 0.5
    max = 5.5
    plt.ylim(min, max)

    y_ticks = np.linspace(min, max, num_ticks)
    plt.yticks(y_ticks)


    plt.grid(linewidth=0.5)

    if not value:
        plt.gca().set_xticklabels([''] * len(plt.gca().get_xticks()))
        plt.gca().set_yticklabels([''] * len(plt.gca().get_yticks()))
    plt.savefig('/home/FedHydro/dataset/code_test/fedsrp/vvv_mae_' + basin_id + '.jpg', dpi=600)
    print('basin:' + basin_id + 'is finished')


if __name__ == '__main__':
    basin03 = '02479155'
    basin01 = '01169000'
    basin18 = '11151300' # 18
    basin17 = '13331500' # 17


    rmse_list3 = [[1.3967109843227332, 1.362851824750565, 1.3457555466751483, 1.298626677826814, 0.961316456381395],
                  [1.36484821887896, 1.4323817205168248, 1.4061719877946144, 1.3777814640123776, 1.2480627945494582],
                  [1.2420828368865313, 1.350467463553767, 1.3138930145891794, 1.3121827496424054, 1.1460932668149608],
                  [1.2179573963958237, 1.4006587085970899, 1.3666009608226874, 1.3497304460822854, 1.0460059305448086],
                  [1.290599292944338, 1.4425647594378512, 1.4095506448338433, 1.469134693319269, 1.1411675449682075],
                  [2.986436319675655, 2.987816427283818, 2.9812949949150025, 2.9224195781893445, 2.4904537297681015]
                  ]
    
    mae_list3 = [    [0.5166058569991201, 0.6722154854690437, 0.6151410087282592, 0.5307346102548176, 0.2060710924968971],
    [0.7453451782882539, 0.8293765649017099, 0.7978764869469441, 0.7173267627146349, 0.2857267442748533],
    [1.061385309877118, 0.9238879165327701, 0.896045699318632, 0.8208611969151881, 0.23009857643417497],
    [1.0146875752604134, 0.9012473063263869, 0.8487111530038313, 0.8095631226411464, 0.2415835809236174],
    [1.8321072763047284, 1.6086599354433864, 1.539704366501977, 1.3370327118226681, 0.5416502499195026],
    [2.109474538620985, 1.891863057689521, 1.8886570770542692, 1.7904383194477123, 1.857562886291552]]

    mae18 = [[0.005133994227788874, 0.0061973884922961716, 0.0061636216345372, 0.006225066853974306, 0.006125796421981505], 
             [0.007175405411860418, 0.009027355484180993, 0.008926160073515791, 0.008980309994401671, 0.008020166818933045], 
             [0.01150583601072295, 0.013782896142326471, 0.013733307631442605, 0.013710547643103022, 0.011262405281725134], 
             [0.01355058937399256, 0.015064632760885628, 0.01513552360129999, 0.015161893951465312, 0.01292540351360991], 
             [0.013869263142903916, 0.01521531614389433, 0.015092738752093875, 0.015096339255298147, 0.012880587443740966], 
             [0.0142919230801704, 0.015336551530049345, 0.015356387602120724, 0.015265664679009265, 0.013428962971720172]]
    
    rmse18 =  [[0.005671109841677511, 0.006848584798314211, 0.006816823836494646, 0.006870127388560781, 0.0067581877230335485], [0.007592110635806187, 0.009563068274398453, 0.009468166063036751, 0.0095209315328278, 0.008363783315336142], [0.011544402013247806, 0.013809787139343372, 0.013760212686185588, 0.013737457055749925, 0.011288698533173479], [0.013561881871559765, 0.015071187845148899, 0.015142807206537445, 0.015171015773618084, 0.01293342585076137], [0.013872924797190654, 0.015221125362685144, 0.015097803016567088, 0.015101149517304872, 0.012882677484693626], [0.014318421387675955, 0.01533789876883319, 0.015357809785385007, 0.015267006049702136, 0.01343818689494816]]


    mae17 =  [[2.7706814068841554, 2.4273925164103294, 2.4194631170699967, 2.469129818690602, 2.8946817964543006], [3.9619198337401063, 3.8594405040771718, 3.872311028528312, 3.759993024030463, 3.8832424825218173], [5.407390164528097, 3.8937449918777594, 3.8794075423559122, 3.700755750826454, 3.9784010581517464], [4.272364147843496, 4.408012577944956, 4.38580028483043, 4.2766182599013, 4.306714882881146], [4.448628804095732, 4.602441238116052, 4.586782207789219, 4.544990983661724, 4.472450591724806], [4.203162722042823, 4.682479051933212, 4.663373555679157, 4.668250518851597, 4.499498138290607]]
 
    rmse171 =  [[3.074816586736001, 2.2488108973200536, 2.309519384610277, 2.3532790261750574, 2.630748282617011], [4.5415910632966305, 3.3374774886910754, 3.377899117362982, 3.2383028198575476, 3.1563749516500272], [4.656738250799446, 3.448893401154565, 3.455643529333386, 3.227337468617614, 3.401790593601225], [4.886901390478501, 3.8370229097668926, 3.847536861515091, 3.721338665469618, 3.472956952188753], [3.8270812218244603, 3.97198216397225, 3.9456103480677585, 3.9051134473360976, 3.737240711744915], [3.0607421947249427, 3.8451156034542873, 3.843590431898653, 3.8139512114050724, 3.5887898097359696]]

    rmse172 =  [[3.1173383194021134, 2.6172829519501404, 2.6190545511183285, 2.6360049173217353, 3.1032114746424577], [3.9842053109119435, 3.8552801922037023, 3.8700631325364308, 3.7616465345885435, 3.9143752748873895], [5.464103855443279, 3.8862705993808437, 3.875645627932387, 3.716669010312808, 3.975813452523057], [4.244811816664801, 4.398072218831134, 4.383743619932395, 4.270021725458263, 4.29180486568193], [4.446976479075986, 4.603084644682193, 4.587238556466922, 4.53826871048745, 4.484591384169848], [4.162948069848652, 4.677215339449627, 4.664046684341977, 4.67021472585715, 4.496699127375652]]


    mae013 =  [[0.3786937935685372, 0.3378325538731017, 0.3400093711888072, 0.32977378964765225, 0.32929881993589366], 
[0.4030742956306205, 0.4382657805552713, 0.438385236769446, 0.42613754614273136, 0.40670469927612407], 
[0.6094726273571838, 0.6228703842359613, 0.6268865202576251, 0.6266509945988239, 0.6400316892464167], 
[0.7566114291087757, 0.7893688122812029, 0.7918141635331563, 0.7930735222277766, 0.7731456337606338], 
[1.4639981016392796, 1.4863351260590956, 1.4894869985504635, 1.492555344964434, 1.4641649150135794], 
[3.5933694443529585, 3.642068369881882, 3.647533819029479, 3.643889496596127, 3.6953281020015614]]


    mae012 =  [[0.2853692714894819, 0.43283374696956706, 0.4332544319875177, 0.386708536457659, 0.2830308397950297], 
[0.3892348690743339, 0.46479906238171437, 0.4660474736728358, 0.41756185792848727, 0.40888257621624546], 
[0.6380365952047178, 0.5338236563999318, 0.5299112164548023, 0.50798666309372, 0.556212202231211], 
[1.360339471736453, 1.3853334137213076, 1.3790608441324141, 1.3438045492194948, 1.2665398849292033], 
[1.4437943033347784, 1.5661939874624702, 1.5621149872289375, 1.556145711035966, 1.5085862816413254], 
[3.3248814537282416, 3.438671125959551, 3.442449138031187, 3.423515453142736, 3.347638241464594]]


    mae011 =  [[1.0830642651586422, 1.2574017451484651, 1.2402159920169387, 1.1429205691318538, 1.1145059868635718], 
[1.8604416672961848, 1.7382878924680247, 1.7499158753673068, 1.7029496067276297, 1.9340144479474992], 
[1.364570256651115, 1.4527751089272771, 1.4082599630306145, 1.349015695127919, 1.8819838100913608], 
[1.5220165637830154, 1.726897319704458, 1.6348498850514266, 1.6265923348565847, 1.9140615627522262], 
[2.9966650502178362, 3.7489191454569335, 3.627249098384966, 3.3755832593811106, 1.8993820555496355], 
[4.509826495117638, 4.20287898868404, 4.143092446333622, 3.8603804389852145, 2.8153666300100784]]


    rmse013 =  [[0.5752155932936468, 0.5903604057228801, 0.5916632659472889, 0.5742281105962058, 0.5973225684638729], [0.6680997168211769, 0.7200451460156556, 0.7227241551935255, 0.707387630096318, 0.7349974561466838], [0.9723946765988326, 0.9953011829926289, 0.9987139015009987, 0.9961482668990429, 1.0533912327674508], [1.449327517253595, 1.4680581357351974, 1.4709612401779668, 1.4686644477704778, 1.4882148401912823], [2.0645757555664312, 2.0720691662430712, 2.0738964565702243, 2.0748748177719656, 2.084538501502979], [4.519652167981237, 4.56909079466861, 4.5722209005187695, 4.566311661880011, 4.656927399688258]]


    rmse012 =  [[0.6728317730750674, 0.6645908674245504, 0.6648415690329981, 0.6424826789259198, 0.5693499994996554], [1.1214944895252168, 1.1549485817931033, 1.1542614713864512, 1.155113426679722, 1.2322515108280894], [1.5801601952155389, 1.6130307831632444, 1.612197369452466, 1.6123209188677374, 1.6904501126202334], [2.2394164566329784, 2.2542694238934904, 2.256013643619852, 2.2589861514130805, 2.3490785977430466], [4.684444840676597, 4.7736646056584355, 4.773604243812524, 4.762517655402166, 4.957768665394733], [6.877174232914109, 6.944976864526375, 6.946540192940928, 6.926960670323867, 7.057634453125066]]

    rmse011 =  [[1.3737740784858812, 1.7924601830194196, 1.7862838553975575, 1.4747467421477884, 0.9143838183219174], [1.6163071968742462, 2.7778119143856763, 2.8410471119380256, 2.635488708898828, 1.5327495503274173], [1.4207513979910629, 3.2971381957269865, 3.0698992462716563, 3.0647431125821756, 1.2991439801871558], [2.093420677607517, 4.887636782044305, 4.526168343562383, 4.833990984592514, 1.8099245235378267], [2.0556473854937276, 2.5929453121962585, 2.5644970099042297, 2.3079868902168683, 1.1988903069302164], [2.984315644435769, 3.7915962372283296, 3.7102179979545786, 3.502298942426473, 2.621734187004629]]

    mae01_a=[[0.5115439142348771, 0.6848135190388184, 0.68551023525898, 0.6336278875031071, 0.39703769427949953], 
[0.7909219782511585, 0.9799976271408455, 0.9725134020186548, 0.9060554118514511, 0.5883405913957782], 
[0.8405027462197114, 1.0260034473900403, 1.0091609494248546, 0.9674462540804096, 0.6891338862619052], 
[0.9584194030849699, 1.0629262008038878, 1.0591383024667378, 1.0161199946569073, 0.8088013530455962], 
[2.885570064673975, 2.262545192025679, 2.2514788789759685, 2.2284186738957916, 2.190418662020741], 
[3.222300891293555, 2.661063883732443, 2.6366970742815234, 2.5572492198113905, 2.4261426314568295]]

    rmse17_a =  [[1.9111026605522636, 3.2804647795900914, 3.2562773386811905, 3.2430529619670203, 2.469334533945842, 3.355579527714057], [5.247115173063873, 4.179377883178315, 4.158863623895507, 4.216932325377278, 4.04673432482731, 6.171463003194648], [6.173356240543622, 4.8738586694420585, 4.897863671133692, 4.760988128028169, 4.383226758165398, 6.958879252386725], [5.560799383229388, 4.974195771756886, 4.9678427688465465, 4.77309504740276, 4.3559923542987375, 6.790323784621072], [5.158408194439293, 4.949709861258463, 4.929450568372615, 4.830272261412814, 4.741570763423228, 6.741720020220286], [5.038655886764126, 5.004892694750449, 5.001009891322284, 4.929396045030991, 4.883997159095554, 6.7677781901852745]]


    mae17_a =  [[1.598771970810111, 2.6670090896286314, 2.6461422272587685, 2.66278973829705, 1.9793452148389497, 2.9818277874500216], [5.184057515497, 4.161327878689598, 4.140313028931158, 4.20024304791299, 4.026504676323425, 6.153202877478899], [6.1248023005219325, 4.86586608697087, 4.889794097538403, 4.754883392196003, 4.3789525406785135, 6.9546319116684225], [5.534410010818189, 4.97124287733573, 4.964769428255828, 4.770288116046013, 4.344161335657941, 6.9546319116684225], [5.153595751409177, 4.948850806556997, 4.928583360965878, 4.82875152797297, 4.740088687563108, 6.74043604155579], [5.0369089527739614, 5.004439571422154, 5.000558895512912, 4.928877062289958, 4.880482040874006, 6.766570894230575]]

    mae03_a = [[1.5605320141340897, 1.1739869473664903, 1.1521300320521939, 1.0494822615230186, 0.8740544655007642, 1.8678975346804316], [1.4420210965725317, 1.511034489513171, 1.4665515241454563, 1.413464204156651, 1.1875920898309063, 2.270112659471535], [1.5573122470878813, 1.7603916834164761, 1.7151983594374345, 1.6795274480846407, 1.373844890696198, 2.5779047288742998], [1.8011335479049084, 1.846707193301393, 1.7915855452079872, 1.7675882590890226, 1.6084354357280795, 2.5243008284864206], [1.789237662773952, 1.707111857800652, 1.6732344865418862, 1.6667193041405028, 1.6108192357314461, 1.9268494275393326], [3.2553955482583263, 3.124182639366321, 3.1063104242954034, 2.9722760073915624, 2.671596476117404, 3.3436380766442784]]



    mae01_b=[[1.7516029330345861, 2.460524362880241, 2.4220840664853394, 2.0140988966755007, 0.9235326190199709, 4.28220788329103], [2.0808049344589046, 2.865079961671729, 2.8299824081162877, 2.3901453501728445, 0.999053214419613, 4.833741253478657], [2.2080230275183546, 2.978197184048954, 2.945012818394516, 2.511348826648572, 1.1448768496705577, 4.973631566488423], [2.5376934602519725, 2.7679143130740558, 2.749523836925054, 2.506046275719643, 1.9937260512139157, 4.985789882279534], [3.179177357114897, 2.8736883531275215, 2.847613432360184, 2.807881153815513, 2.3609890747150937, 5.031397949011623], [3.3286338218707074, 3.1604182610036693, 3.1452972384240954, 3.158022152127155, 3.0262892004563167, 4.0058657857429045]]

    mae18_a = [[0.0025086870953789264, 0.0026003561482742217, 0.0025873839488151805, 0.0026018459144450425, 0.002086672543853619, 0.003032005209608964], [0.004767827926320346, 0.004974910936288415, 0.004975778517432124, 0.004972347987466547, 0.004353007793743839, 0.005407715353122046], [0.0050301712755974545, 0.005378045808519294, 0.005380551789236328, 0.0053772005579387164, 0.004546523128935898, 0.005768714119391194], [0.004959365205453086, 0.0053079146334672565, 0.005312857394146377, 0.005331173329972368, 0.00468381873266885, 0.005665037861286566], [0.004972489060195114, 0.005312532146278119, 0.0053129877856484065, 0.005314203466409198, 0.004653529534288727, 0.0057556751697998626], [0.005267563627963761, 0.005322015841467463, 0.005322496734386789, 0.005332438713797045, 0.004789660247293974, 0.005801202713287082]]

    rmse01_a = [[0.5974218316683778, 0.9428303029818962, 0.9151409867766536, 0.7445695473797698, 0.46029451790833853], [1.1068932138055712, 1.4407888281243921, 1.4303599804254434, 1.21534092186569, 0.609669637498539], [1.5476658320334513, 1.6909769053276584, 1.636894133298166, 1.4532659525045584, 1.0432295627821273], [1.2670244311507117, 1.715376724840595, 1.6716187954999178, 1.5035303137865155, 0.6747230340049484], [1.4793563781180084, 1.6719308958015662, 1.659568954196385, 1.4870584235212314, 0.6507949545586948], [5.207660696199601, 4.42119889233129, 4.409462128917145, 4.324493794445591, 4.296310410882922]]

    rmse18_h = [[0.0026961606501775927, 0.0028235836865135077, 0.0028227755101528106, 0.0028279824703380148, 0.0023244144348295677, 0.0032234158542701224],
[0.004796176026613511, 0.004984483368283995, 0.004985545362560549, 0.004982449904494271, 0.00438222518939893, 0.005418699352790099],
[0.005045579597942963, 0.005374412510202507, 0.005380231934063544, 0.005380370875707722, 0.004542003842691103, 0.0057688851490128755],
[0.00494784763343, 0.005320386044687531, 0.005314259948478016, 0.00533603976796719, 0.004730122308564168, 0.005665541043359433],
[0.004980797557222889, 0.0053153615148828475, 0.005313229756529512, 0.005320237961190996, 0.004648342907937267, 0.005755853069345786],
[0.005266566081122181, 0.005323842263479799, 0.00532952869571255, 0.005330970455812675, 0.004836428930926511, 0.00580153717661339]]


    # mk_nse_pic(basin_id, nse_list_03)
    mk_rmse_pic(basin18, rmse18_h, 0)
    #mk_mae_pic(basin01, mae01_b, 0)