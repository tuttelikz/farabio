Search.setIndex({docnames:["handbook/changelog","handbook/concepts","handbook/downloads","handbook/faq","handbook/index","handbook/overview","handbook/tutorial","index","installation","reference/core.basetrainer","reference/core.configs","reference/core.convnettrainer","reference/core.gantrainer","reference/data.dirops","reference/data.imgops","reference/index","reference/models.detection.faster_rcnn.faster_rcnn_trainer","reference/models.detection.yolov3.yolo_trainer","reference/models.segmentation.attunet.attunet_trainer","reference/models.segmentation.unet.unet_trainer","reference/models.superres.srgan.srgan_trainer","reference/models.translation.cyclegan.cyclegan_trainer"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":3,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["handbook/changelog.rst","handbook/concepts.rst","handbook/downloads.rst","handbook/faq.rst","handbook/index.rst","handbook/overview.rst","handbook/tutorial.rst","index.rst","installation.rst","reference/core.basetrainer.rst","reference/core.configs.rst","reference/core.convnettrainer.rst","reference/core.gantrainer.rst","reference/data.dirops.rst","reference/data.imgops.rst","reference/index.rst","reference/models.detection.faster_rcnn.faster_rcnn_trainer.rst","reference/models.detection.yolov3.yolo_trainer.rst","reference/models.segmentation.attunet.attunet_trainer.rst","reference/models.segmentation.unet.unet_trainer.rst","reference/models.superres.srgan.srgan_trainer.rst","reference/models.translation.cyclegan.cyclegan_trainer.rst"],objects:{"farabi.core.basetrainer":{BaseTrainer:[9,0,1,""]},"farabi.core.basetrainer.BaseTrainer":{build_model:[9,1,1,""],evaluate:[9,1,1,""],get_testloader:[9,1,1,""],get_trainloader:[9,1,1,""],init_attr:[9,1,1,""],test:[9,1,1,""],train:[9,1,1,""]},"farabi.core.configs":{_cfg_attunet:[10,2,1,""],_cfg_cyclegan:[10,2,1,""],_cfg_fasterrcnn:[10,2,1,""],_cfg_srgan:[10,2,1,""],_cfg_unet:[10,2,1,""],_cfg_yolov3:[10,2,1,""]},"farabi.core.convnettrainer":{ConvnetTrainer:[11,0,1,""]},"farabi.core.convnettrainer.ConvnetTrainer":{__init__:[11,1,1,""],build_model:[11,1,1,""],build_parallel_model:[11,1,1,""],default_attr:[11,1,1,""],define_compute_attr:[11,1,1,""],define_data_attr:[11,1,1,""],define_log_attr:[11,1,1,""],define_misc_attr:[11,1,1,""],define_model_attr:[11,1,1,""],define_test_attr:[11,1,1,""],define_train_attr:[11,1,1,""],evaluate_batch:[11,1,1,""],evaluate_epoch:[11,1,1,""],exit_trainer:[11,1,1,""],get_testloader:[11,1,1,""],get_trainloader:[11,1,1,""],init_attr:[11,1,1,""],load_model:[11,1,1,""],load_parallel_model:[11,1,1,""],loss_backward:[11,1,1,""],on_end_test_batch:[11,1,1,""],on_end_training_batch:[11,1,1,""],on_epoch_end:[11,1,1,""],on_evaluate_batch_end:[11,1,1,""],on_evaluate_batch_start:[11,1,1,""],on_evaluate_end:[11,1,1,""],on_evaluate_epoch_end:[11,1,1,""],on_evaluate_epoch_start:[11,1,1,""],on_evaluate_start:[11,1,1,""],on_start_test_batch:[11,1,1,""],on_start_training_batch:[11,1,1,""],on_test_end:[11,1,1,""],on_test_start:[11,1,1,""],on_train_end:[11,1,1,""],on_train_epoch_end:[11,1,1,""],on_train_epoch_start:[11,1,1,""],on_train_start:[11,1,1,""],optimizer_step:[11,1,1,""],optimizer_zero_grad:[11,1,1,""],save_model:[11,1,1,""],save_parallel_model:[11,1,1,""],start_logger:[11,1,1,""],stop_train:[11,1,1,""],test:[11,1,1,""],test_loop:[11,1,1,""],test_step:[11,1,1,""],train:[11,1,1,""],train_batch:[11,1,1,""],train_epoch:[11,1,1,""],train_loop:[11,1,1,""],training_step:[11,1,1,""]},"farabi.core.gantrainer":{GanTrainer:[12,0,1,""]},"farabi.core.gantrainer.GanTrainer":{__init__:[12,1,1,""],build_model:[12,1,1,""],default_attr:[12,1,1,""],define_compute_attr:[12,1,1,""],define_data_attr:[12,1,1,""],define_log_attr:[12,1,1,""],define_misc_attr:[12,1,1,""],define_model_attr:[12,1,1,""],define_test_attr:[12,1,1,""],define_train_attr:[12,1,1,""],discriminator_backward:[12,1,1,""],discriminator_loss:[12,1,1,""],discriminator_optim_step:[12,1,1,""],discriminator_zero_grad:[12,1,1,""],evaluate_batch:[12,1,1,""],evaluate_epoch:[12,1,1,""],generator_backward:[12,1,1,""],generator_loss:[12,1,1,""],generator_optim_step:[12,1,1,""],generator_zero_grad:[12,1,1,""],get_dataloader:[12,1,1,""],get_testloader:[12,1,1,""],get_trainloader:[12,1,1,""],init_attr:[12,1,1,""],load_model:[12,1,1,""],on_end_test_batch:[12,1,1,""],on_end_training_batch:[12,1,1,""],on_epoch_end:[12,1,1,""],on_evaluate_batch_end:[12,1,1,""],on_evaluate_batch_start:[12,1,1,""],on_evaluate_end:[12,1,1,""],on_evaluate_epoch_end:[12,1,1,""],on_evaluate_epoch_start:[12,1,1,""],on_evaluate_start:[12,1,1,""],on_start_test_batch:[12,1,1,""],on_start_training_batch:[12,1,1,""],on_test_end:[12,1,1,""],on_test_start:[12,1,1,""],on_train_end:[12,1,1,""],on_train_epoch_end:[12,1,1,""],on_train_epoch_start:[12,1,1,""],on_train_start:[12,1,1,""],save_model:[12,1,1,""],start_logger:[12,1,1,""],stop_train:[12,1,1,""],test:[12,1,1,""],test_loop:[12,1,1,""],test_step:[12,1,1,""],train:[12,1,1,""],train_batch:[12,1,1,""],train_epoch:[12,1,1,""],train_loop:[12,1,1,""]},"farabi.data.dirops":{DirOps:[13,0,1,""]},"farabi.data.dirops.DirOps":{del_files:[13,1,1,""],dirinfo:[13,1,1,""],lsmedia:[13,1,1,""],split_traintest:[13,1,1,""]},"farabi.data.imgops":{ImgOps:[14,0,1,""]},"farabi.data.imgops.ImgOps":{approx_bcg:[14,1,1,""],blend_img:[14,1,1,""],blend_imgs:[14,1,1,""],get_date:[14,1,1,""],mask_img:[14,1,1,""],pad_img:[14,1,1,""],print_imginfo:[14,1,1,""],profile_img:[14,1,1,""],slice_img:[14,1,1,""]},"farabi.models.detection.faster_rcnn.faster_rcnn_trainer":{FasterRCNNTrainer:[16,0,1,""]},"farabi.models.detection.faster_rcnn.faster_rcnn_trainer.FasterRCNNTrainer":{build_model:[16,1,1,""],define_log_attr:[16,1,1,""],define_misc_attr:[16,1,1,""],define_model_attr:[16,1,1,""],define_train_attr:[16,1,1,""],evaluate_batch:[16,1,1,""],forward:[16,1,1,""],get_meter_data:[16,1,1,""],get_testloader:[16,1,1,""],get_trainloader:[16,1,1,""],load_model:[16,1,1,""],loss_backward:[16,1,1,""],on_epoch_end:[16,1,1,""],on_evaluate_batch_start:[16,1,1,""],on_evaluate_epoch_end:[16,1,1,""],on_evaluate_epoch_start:[16,1,1,""],on_start_training_batch:[16,1,1,""],on_train_epoch_start:[16,1,1,""],on_train_start:[16,1,1,""],optimizer_step:[16,1,1,""],optimizer_zero_grad:[16,1,1,""],reset_meters:[16,1,1,""],save:[16,1,1,""],save_model:[16,1,1,""],start_logger:[16,1,1,""],training_step:[16,1,1,""],update_meters:[16,1,1,""],visdom_plot:[16,1,1,""]},"farabi.models.detection.yolov3.yolo_trainer":{YoloTrainer:[17,0,1,""]},"farabi.models.detection.yolov3.yolo_trainer.YoloTrainer":{build_model:[17,1,1,""],define_compute_attr:[17,1,1,""],define_data_attr:[17,1,1,""],define_log_attr:[17,1,1,""],define_misc_attr:[17,1,1,""],define_model_attr:[17,1,1,""],define_test_attr:[17,1,1,""],define_train_attr:[17,1,1,""],detect_perform:[17,1,1,""],evaluate_batch:[17,1,1,""],get_detections:[17,1,1,""],get_trainloader:[17,1,1,""],on_end_training_batch:[17,1,1,""],on_epoch_end:[17,1,1,""],on_evaluate_batch_start:[17,1,1,""],on_evaluate_epoch_end:[17,1,1,""],on_evaluate_epoch_start:[17,1,1,""],on_start_training_batch:[17,1,1,""],on_train_epoch_start:[17,1,1,""],plot_bbox:[17,1,1,""],save_model:[17,1,1,""],test:[17,1,1,""],training_step:[17,1,1,""]},"farabi.models.segmentation.attunet.attunet_trainer":{AttunetTrainer:[18,0,1,""]},"farabi.models.segmentation.attunet.attunet_trainer.AttunetTrainer":{build_model:[18,1,1,""],build_parallel_model:[18,1,1,""],define_compute_attr:[18,1,1,""],define_data_attr:[18,1,1,""],define_log_attr:[18,1,1,""],define_misc_attr:[18,1,1,""],define_model_attr:[18,1,1,""],define_test_attr:[18,1,1,""],define_train_attr:[18,1,1,""],evaluate_batch:[18,1,1,""],generate_result_img:[18,1,1,""],get_testloader:[18,1,1,""],get_trainloader:[18,1,1,""],load_model:[18,1,1,""],load_parallel_model:[18,1,1,""],loss_backward:[18,1,1,""],on_end_test_batch:[18,1,1,""],on_end_training_batch:[18,1,1,""],on_epoch_end:[18,1,1,""],on_evaluate_batch_end:[18,1,1,""],on_evaluate_epoch_end:[18,1,1,""],on_evaluate_epoch_start:[18,1,1,""],on_start_training_batch:[18,1,1,""],on_test_start:[18,1,1,""],on_train_epoch_end:[18,1,1,""],on_train_epoch_start:[18,1,1,""],optimizer_step:[18,1,1,""],optimizer_zero_grad:[18,1,1,""],save_model:[18,1,1,""],save_parallel_model:[18,1,1,""],show_model_summary:[18,1,1,""],start_logger:[18,1,1,""],test_step:[18,1,1,""],training_step:[18,1,1,""]},"farabi.models.segmentation.unet.unet_trainer":{UnetTrainer:[19,0,1,""]},"farabi.models.segmentation.unet.unet_trainer.UnetTrainer":{build_model:[19,1,1,""],build_parallel_model:[19,1,1,""],define_compute_attr:[19,1,1,""],define_data_attr:[19,1,1,""],define_log_attr:[19,1,1,""],define_misc_attr:[19,1,1,""],define_model_attr:[19,1,1,""],define_test_attr:[19,1,1,""],define_train_attr:[19,1,1,""],evaluate_batch:[19,1,1,""],generate_result_img:[19,1,1,""],get_testloader:[19,1,1,""],get_trainloader:[19,1,1,""],load_model:[19,1,1,""],load_parallel_model:[19,1,1,""],loss_backward:[19,1,1,""],on_end_test_batch:[19,1,1,""],on_end_training_batch:[19,1,1,""],on_epoch_end:[19,1,1,""],on_evaluate_batch_end:[19,1,1,""],on_evaluate_epoch_end:[19,1,1,""],on_evaluate_epoch_start:[19,1,1,""],on_start_training_batch:[19,1,1,""],on_test_start:[19,1,1,""],on_train_epoch_end:[19,1,1,""],on_train_epoch_start:[19,1,1,""],optimizer_step:[19,1,1,""],optimizer_zero_grad:[19,1,1,""],save_model:[19,1,1,""],save_parallel_model:[19,1,1,""],show_model_summary:[19,1,1,""],start_logger:[19,1,1,""],test_step:[19,1,1,""],training_step:[19,1,1,""]},"farabi.models.superres.srgan.srgan_trainer":{SrganTrainer:[20,0,1,""]},"farabi.models.superres.srgan.srgan_trainer.SrganTrainer":{build_model:[20,1,1,""],define_compute_attr:[20,1,1,""],define_data_attr:[20,1,1,""],define_log_attr:[20,1,1,""],define_misc_attr:[20,1,1,""],define_model_attr:[20,1,1,""],define_train_attr:[20,1,1,""],discriminator_backward:[20,1,1,""],discriminator_loss:[20,1,1,""],discriminator_optim_step:[20,1,1,""],discriminator_zero_grad:[20,1,1,""],evaluate_batch:[20,1,1,""],generator_backward:[20,1,1,""],generator_loss:[20,1,1,""],generator_optim_step:[20,1,1,""],generator_zero_grad:[20,1,1,""],get_testloader:[20,1,1,""],get_trainloader:[20,1,1,""],load_model:[20,1,1,""],on_end_training_batch:[20,1,1,""],on_epoch_end:[20,1,1,""],on_evaluate_batch_end:[20,1,1,""],on_evaluate_epoch_end:[20,1,1,""],on_evaluate_epoch_start:[20,1,1,""],on_start_training_batch:[20,1,1,""],on_test_end:[20,1,1,""],on_test_start:[20,1,1,""],on_train_epoch_start:[20,1,1,""],optimizer_zero_grad:[20,1,1,""],save_csv:[20,1,1,""],save_model:[20,1,1,""],start_logger:[20,1,1,""],test_batch:[20,1,1,""],test_step:[20,1,1,""],train_batch:[20,1,1,""]},"farabi.models.translation.cyclegan.cyclegan_trainer":{CycleganTrainer:[21,0,1,""]},"farabi.models.translation.cyclegan.cyclegan_trainer.CycleganTrainer":{build_model:[21,1,1,""],cycle_g_loss:[21,1,1,""],define_compute_attr:[21,1,1,""],define_data_attr:[21,1,1,""],define_log_attr:[21,1,1,""],define_misc_attr:[21,1,1,""],define_model_attr:[21,1,1,""],define_test_attr:[21,1,1,""],define_train_attr:[21,1,1,""],discriminatorA_backward:[21,1,1,""],discriminatorA_loss:[21,1,1,""],discriminatorA_optim_step:[21,1,1,""],discriminatorA_zero_grad:[21,1,1,""],discriminatorB_backward:[21,1,1,""],discriminatorB_loss:[21,1,1,""],discriminatorB_optim_step:[21,1,1,""],discriminatorB_zero_grad:[21,1,1,""],fake_dA_loss:[21,1,1,""],fake_dB_loss:[21,1,1,""],gan_g_loss:[21,1,1,""],generator_backward:[21,1,1,""],generator_loss:[21,1,1,""],generator_optim_step:[21,1,1,""],generator_zero_grad:[21,1,1,""],get_testloader:[21,1,1,""],get_trainloader:[21,1,1,""],identity_g_loss:[21,1,1,""],load_model:[21,1,1,""],on_end_test_batch:[21,1,1,""],on_end_training_batch:[21,1,1,""],on_epoch_end:[21,1,1,""],on_start_training_batch:[21,1,1,""],on_test_start:[21,1,1,""],on_train_epoch_end:[21,1,1,""],on_train_epoch_start:[21,1,1,""],on_train_start:[21,1,1,""],real_dA_loss:[21,1,1,""],real_dB_loss:[21,1,1,""],save_model:[21,1,1,""],start_logger:[21,1,1,""],test_step:[21,1,1,""],train_batch:[21,1,1,""]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:method","2":"py:function"},terms:{"01497":16,"02640":17,"02767":17,"03999":18,"04597":19,"04802":20,"100":14,"1024":14,"10593":21,"1505":19,"1506":[16,17],"1609":20,"1703":21,"1804":[17,18],"200820_191645":14,"2013":2,"2017":2,"2018":2,"2019":2,"2021":0,"6006":6,"abstract":[9,11,12,16,17,18,19,21],"class":[5,6,9,13,14],"default":13,"float":[13,14],"function":[4,5,10,11,12,14],"import":[6,13,14],"int":[11,12,13,14,20,21],"new":0,"public":4,"return":[13,14,21],"short":5,"static":14,"super":[2,5,20],"true":13,The:[5,10,16],There:[0,5],These:10,Using:20,Will:6,__init__:[11,12],__main__:6,__name__:6,_cfg_attunet:10,_cfg_cyclegan:10,_cfg_fasterrcnn:10,_cfg_srgan:10,_cfg_unet:10,_cfg_yolov3:10,abbrevi:5,abc:[9,11],abl:5,abov:16,abs:[16,17,18,19,20,21],access:[2,11,12],accord:6,accur:5,action:[11,12,18,19,20,21],activ:4,add:0,adding:0,adjust:6,adversari:[20,21],aitorzip:21,albument:8,all:[0,5,8,9],analyz:5,answer:3,append:5,applic:5,approach:5,approx_bcg:14,approxim:14,apto:2,architectur:[11,16],area:14,arg:[0,9,11,12,16,17,18,19,20,21],argument:5,arrai:[1,14],arxiv:[16,17,18,19,20,21],askaruli:0,assert:6,associ:5,attent:[10,18],attribut:[13,14,16,17,18,19,20,21],attunet:[5,6,18],attunet_train:[5,6,15],attunettrain:[5,6],aud_fn:13,audio:13,autopep8:8,averag:14,avoid:5,b08457c770:0,back:[11,12,16,18,19,20],background:14,backward:[12,20,21],base:[9,10],basetrain:[5,11,15,17,18,19,21],bash:6,basic:[13,14],batch:[11,12,16,17,18,19,20,21],bboxtool:5,befor:13,begin:1,benchmark:2,better:5,between:14,binari:14,biomed:[0,4,5],blend:14,blend_img:14,blind:2,block:5,blue:14,bool:13,border:5,both:9,bowl:2,branch:0,bug:5,build:[9,11,12,16,17,18,19,20,21],build_model:[9,11,12,16,17,18,19,20,21],build_parallel_model:[11,12,18,19],can:[4,11,12],cancer:2,cannot:18,capit:5,capword:5,cervic:2,cfg:[5,6],challeng:2,chang:[4,6],changelog:[4,7],channel:14,chenyuntc:16,chest:2,choos:6,clash:5,class_:5,classfic:5,classif:16,clss:5,cnn:16,coco:[2,5],code:[1,6,16,17,18,19,20,21],col1:14,col2:14,collect:[2,11,12],collect_env:5,colour:14,com:[2,8,16,17,18,19,20,21],combin:10,commit:0,competit:2,compil:0,complet:6,complex:5,comput:[11,12,17,18,19,20,21],compute_arg:10,concept:[4,7],conda:3,config:[0,5,6,11,12,15,16,17,18,19,20,21],configur:[0,6],consist:21,constant:5,contain:[10,14],contribut:4,conv:6,convent:5,convnet:11,convnettrain:[5,15,17,18,19],coolenv:8,coordin:14,core:[5,6,7,9,10,11,12,21],corrupt:5,creat:[6,8,13,14],create_custom_model:[5,6],creator_tool:5,cudatoolkit:8,current:[11,12,13,20,21],custom:[4,5,11,12,14,16,17,18,19,20,21],cut:14,cwd:13,cycl:21,cycle_g_loss:21,cyclegan:[2,5,6,10,21],cyclegan_train:[5,6,15],cyclegantrain:[5,6],dai:5,darknet53:6,darknet:5,data:[2,5,6,7,9,10,11,12,13,14,16,17,18,19,20,21],data_config:6,dataload:[5,9,11,12,16,17,18,19,20,21],dataset:[0,4,5,9],date:[8,14],deep:5,default_attr:[11,12],default_cfg:6,defin:[5,9,10,11,12,16,17,18,19,20,21],define_compute_attr:[11,12,17,18,19,20,21],define_data_attr:[11,12,17,18,19,20,21],define_log_attr:[11,12,16,17,18,19,20,21],define_misc_attr:[11,12,16,17,18,19,20,21],define_model_attr:[11,12,16,17,18,19,20,21],define_test_attr:[11,12,17,18,19,21],define_train_attr:[11,12,16,17,18,19,20,21],definit:14,del_fil:13,delet:[0,13],dermatoscop:2,desir:14,detect:[2,5,6,16,17],detect_perform:[6,17],dict:0,dictionari:0,digit:5,dimens:14,dir:[6,13],dir_path:13,dirinfo:13,dirop:[5,15],dirsiz:13,discrimin:[12,20,21],discriminator_backward:[12,20],discriminator_loss:[12,20],discriminator_optim_step:[12,20],discriminator_zero_grad:[12,20],discriminatora_backward:21,discriminatora_loss:21,discriminatora_optim_step:21,discriminatora_zero_grad:21,discriminatorb_backward:21,discriminatorb_loss:21,discriminatorb_optim_step:21,discriminatorb_zero_grad:21,div2k:2,doc:0,download:[4,7],droi:14,dure:[11,16,17,18,19],easydict:6,edict:6,edit:8,educ:5,egg:0,either:[11,12],elif:6,end:[1,11,12,16,17,18,19,20,21],engin:5,enhanc:5,ensur:8,entri:[11,12],env:0,epoch:[11,12,16,17,18,19,20,21],equip:5,eriklindernoren:17,especi:5,etim:6,evalu:[2,9,16,17,18,19,20],evaluate_batch:[11,12,16,17,18,19,20],evaluate_epoch:[11,12],everi:11,exact:[11,12],exampl:[13,14],except:[5,13],exist:13,exit:11,exit_train:11,fake:21,fake_da_loss:21,fake_db_loss:21,fals:13,faq:[4,7],farabi:[2,5,6,9,10,11,12,13,14,16,17,18,19,20,21],faster:[10,16],faster_rcnn:[5,6,16],faster_rcnn_train:[5,6,15],faster_rcnn_vgg16:5,fasterrcnn:16,fasterrcnntrain:[5,6],field:6,fig:14,figur:14,file:[6,13,14],filenam:13,find:5,fire:8,first:14,fit:18,flag:[8,13],folder:13,follow:[5,6],format:[13,14],forward:16,frac:1,from:[2,3,5,6,13,18,19],from_loc:8,from_pip:8,fsize:14,fundu:2,gan:21,gan_g_loss:21,gantrain:[5,15,20,21],gener:[5,12,18,19,20,21],generate_result_img:[18,19],generator_backward:[12,20,21],generator_loss:[12,20,21],generator_optim_step:[12,20,21],generator_zero_grad:[12,20,21],get:14,get_dat:14,get_dataload:12,get_detect:17,get_meter_data:16,get_testload:[9,11,12,16,18,19,20,21],get_trainload:[9,11,12,16,17,18,19,20,21],git:[5,16,17,18,19,20,21],github:[2,3,8,16,17,18,19,20,21],going:16,gpu:[3,11,18,19],grad:20,gradient:[11,12,16,18,19,20,21],grai:14,green:14,group:0,ham10000:2,handbook:7,has:5,have:5,head:16,height:14,helper:[5,6],here:[11,12,16,17,18,19,20,21],hire:1,histogram:14,histopatholog:[2,5],hook:[16,17,18,19,20,21],hooksa:11,horizont:14,hour:14,how:[3,4,7],howev:17,http:[2,8,16,17,18,19,20,21],human:14,iccv:2,ident:21,identity_g_loss:21,imag:[2,5,13,14,18,19,21],image_segment:18,imageio:8,img1:14,img2:14,img:14,img_b:14,img_blend:14,img_fn:13,img_g:14,img_mask:14,img_ov:14,img_pad:14,img_path:14,img_r:14,img_ref:14,img_slic:14,img_slices_info:14,imgmask:14,imgop:[5,13,15],imgpath:14,imgtoblend:14,implement:[16,17,18,19,20,21],improv:5,includ:[13,16],index:7,info:14,inform:[13,14],inherit:[9,11,17,18,19,21],init:6,init_attr:[9,11,12],initi:[9,11,12],instal:[3,7],instanc:[5,13,14],integr:5,intel:2,intens:14,interact:14,interest:[5,13,14],ipdb:8,issu:3,item:13,itim:6,jpg:14,kaggl:2,kernel:[5,6],keyword:5,kitti:2,know:3,kwarg:16,l1loss:21,lab:5,lane:2,larg:2,learn:5,leejunhyun:18,left:1,leftthoma:20,leq:1,lesion:2,letter:5,level:5,lifecycl:17,link:0,linux:4,list:[6,8,13,14],lll:1,load:[6,11,12,16,18,19,20,21],load_model:[11,12,16,18,19,20,21],load_parallel_model:[11,18,19],local:[8,16],log:[11,12,16,17,18,19,20,21],log_arg:10,logdir:6,logger:[5,11,12,16,18,19,20,21],loop:[9,16,17,18,19,20,21],loss:[5,11,16,18,19,21],loss_backward:[11,16,18,19],lowercas:5,lsmedia:13,main:[0,10,11],make:[11,12],mask:14,mask_img:14,match:13,matplotlib:[8,14],max:[1,14],media:13,medic:5,memori:13,meter:5,method:[9,13,14,16,17,18,19,20,21],metric:5,milesi:19,min:[1,14],misc:5,misc_arg:10,miscellan:[11,12,16,17,18,19,20,21],mnist:2,mobileodt:2,mode:[6,8,18],model:[0,3,4,5,7,9,11,12,16,17,18,19,20,21],model_def:6,model_nam:20,modifi:6,modul:[5,7,15],monet2photo:2,monitor:5,mseloss:21,multi:[2,11,18,19],name:[5,6],nano:6,navig:6,ndir:13,neat:13,necessari:5,neq:1,net:[10,18,19],network:[16,20,21],nfile:13,none:14,normal:5,notimplementederror:9,ntire:2,nuclei:2,num:6,number:[13,14],numpi:[8,14],numpydoc:0,object:[2,11,12,14,17,20],occupi:13,on_end_test_batch:[11,12,18,19,21],on_end_training_batch:[11,12,17,18,19,20,21],on_epoch_end:[11,12,16,17,18,19,20,21],on_evaluate_batch_end:[11,12,18,19,20],on_evaluate_batch_start:[11,12,16,17],on_evaluate_end:[11,12],on_evaluate_epoch_end:[11,12,16,17,18,19,20],on_evaluate_epoch_start:[11,12,16,17,18,19,20],on_evaluate_start:[11,12],on_start_test_batch:[11,12],on_start_training_batch:[11,12,16,17,18,19,20,21],on_test_end:[11,12,20],on_test_start:[11,12,18,19,20,21],on_train_end:[11,12],on_train_epoch_end:[11,12,18,19,21],on_train_epoch_start:[11,12,16,17,18,19,20,21],on_train_start:[11,12,16,21],one:[18,19],onli:[6,13,18],open:5,opencv_python:8,oper:13,opt:14,optic:5,optim:[11,12,16,18,19,20,21],optimizer_step:[11,16,18,19],optimizer_zero_grad:[11,16,18,19,20],option:14,org:[16,17,18,19,20,21],organ:4,orien:14,orient:14,origin:[16,17,18,19,20,21],our:5,overag:14,overlai:14,overlap:14,overrid:[9,11,12,16,17,18,19,20,21],overview:4,packag:[3,4,7,10],pad:14,pad_img:14,page:7,panda:8,paper:2,parallel:[11,18,19],paramet:[11,12,13,14,17,18,19,20,21],parent:20,parser:5,path:[13,14],pdf:1,pep:5,perform:13,perhap:5,photographi:2,piec:14,pigment:2,pillow:8,pip:3,pipelin:5,pixel:14,place:10,pleas:5,plot:14,plot_bbox:17,pmax:14,pmin:14,pneumonia:2,png:[1,13,14],point:14,port:6,pre:5,predefin:14,prepar:5,prerequisit:7,pretrain:4,pretrained_weight:6,print:[6,13,14],print_imginfo:14,problem:5,process:5,profil:14,profile_img:14,project:14,propag:[11,12,16,18,19,20],properti:9,propos:[16,17,18,19,20,21],prototyp:5,provid:[11,12,13,14],pt1:14,pt2:14,pth:6,pull:5,purpos:4,pursu:5,put:[11,12,18,19,20,21],pypi:8,python:[3,6,8,9],pytorch:[8,16,17,19,21],qualiti:5,quick:[11,12],rai:2,rais:9,rang:14,rapidli:5,rarr:5,rather:5,ratio:[13,14],rcnn:[10,16],readabl:[5,14],real:[17,21],real_da_loss:21,real_db_loss:21,recent:4,recommonmark:8,red:14,ref_:14,refer:7,reflist:13,region:[5,16],region_proposal_network:5,regul:5,relat:[4,10,11,12,16,17,18,19,20,21],releas:2,repres:14,request:5,requir:[0,8],research:5,reserv:5,reset_met:16,resolut:[2,5,20],result:5,retina:2,retreiv:[11,12,16,17,18,19,20,21],retriev:13,reus:5,rgb:14,right:1,road:2,roi:14,roi_cls_loss:16,roi_loc_loss:16,row1:14,row2:14,rpn:16,rpn_cls_loss:16,rpn_loc_loss:16,rtd:0,same:5,sampl:14,sample_list:13,san:0,save:[11,12,16,17,18,19,20,21],save_csv:20,save_dict:16,save_model:[11,12,16,17,18,19,20,21],save_parallel_model:[11,18,19],save_path:16,scalar:21,scale:2,scienc:2,scikit_imag:8,scipi:8,screen:[2,5],seaborn:8,search:7,second:14,segment:[2,5,6,18,19],self:[13,14],send:[5,12,20,21],separ:[5,10],set:[6,10,11,12,16,17,18,19,20,21],setuptool:8,sever:5,shape:14,should:5,show_model_summari:[18,19],shuffl:13,simg:14,simpl:16,singl:5,size:14,skin:2,slice:14,slice_img:14,solut:3,sourc:[1,2,9,10,11,12,13,14,16,17,18,19,20,21],spell:5,sphinx:[0,8],split:13,split_traintest:13,srgan:[5,6,10,20],srgan_train:[5,6,15],srgantrain:[5,6],stackoverflow:3,start:[6,11,12,16,17,18,19,20,21],start_logg:[11,12,16,18,19,20,21],step:[11,12,16,18,19,20,21],stop_train:[11,12],store:6,str:[13,14],subdirectori:13,suit:2,sum:16,superr:[5,6,20],support:18,synonym:5,tabul:8,tag:2,take:14,taken:2,tan:1,tbl:8,tbldata:5,techniqu:5,tensorboard:4,tensorboardx:8,tensorflow:8,terminalt:8,test:[5,6,9,10,13,16,17,18,19,20,21],test_arg:[10,20],test_batch:20,test_list:13,test_loop:[11,12],test_step:[11,12,18,19,20,21],tfrecord:6,than:5,theme:0,thi:[0,4,9,11,12,14,16,17,18,19,20,21],thu:5,time:[6,17],tini:5,titem:13,torch:[8,9,11,12,16,17,18,19,20,21],torchaudio:8,torchsummari:8,torchvis:8,total:[13,21],total_loss:16,tqdm:8,trail:5,train:[4,5,9,13,16,17,18,19,20,21],train_batch:[11,12,20,21],train_epoch:[11,12],train_list:13,train_loop:[11,12],trainer:[6,9,11,12,16,17,18,19,20,21],training_step:[11,16,17,18,19],transfer:2,transform:5,translat:[5,6,21],trnr:6,tupl:[13,14],tutori:[4,5,7],tuttelikz:2,two:14,txt:8,type:9,uncommit:0,underscor:5,unet:[5,6,19],unet_train:[5,6,15],unettrain:[5,6],unist:8,unpair:21,update_met:16,upgrad:8,usag:7,use:[4,5,11,12],used:5,uses:[3,16,17,18,19,20,21],using:[2,5,21],usual:5,util:[5,6,9,11,12,16,17,18,19,20,21],valid:[6,9],valu:14,variabl:5,vertic:14,vgg:5,vid_fn:13,video:13,visdom:8,visdom_plot:16,vision:2,vistool:5,visual:5,wai:[5,11,12],webpag:2,weight:[4,6],weights_path:6,were:0,what:4,wheel:8,when:0,where:6,which:[3,11,12,13,16,17,18,19,20,21],whose:5,width:14,word:5,work:[5,6,16,17,18,19,20,21],written:5,yolo:[4,17],yolo_train:[5,6,15],yolo_v3:5,yolotrain:[5,6],yolov3:[5,6,17],you:5,zero:[11,12,16,18,19,20,21]},titles:["Changelog","Concepts","Downloads","FAQ","Handbook","Overview","Tutorial","farabi","Usage","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">basetrainer</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">configs</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">convnettrainer</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">gantrainer</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">dirops</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">imgops</span></code> Module","Reference","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">faster_rcnn_trainer</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">yolo_trainer</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">attunet_trainer</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">unet_trainer</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">srgan_trainer</span></code> Module","<code class=\"xref py py-mod docutils literal notranslate\"><span class=\"pre\">cyclegan_trainer</span></code> Module"],titleterms:{"class":[11,12,16,17,18,19,20,21],"default":[10,11,12],"function":1,"public":2,activ:[1,8],arctan:1,arctang:1,argument:10,attribut:[11,12],attunet_train:18,attunettrain:18,basetrain:9,binari:1,biomed:2,bipolar:1,can:5,categori:10,caveat:5,chang:0,changelog:0,clean:5,clone:8,code:5,concept:1,conda:8,config:10,configur:10,content:[1,6],contribut:5,convnettrain:11,core:15,custom:6,cyclegan_train:21,cyclegantrain:21,data:15,dataset:[2,6],diagram:5,dirop:13,doc:[11,12],download:2,elu:1,environ:8,evalu:[11,12],exponenti:1,faq:3,farabi:7,faster_rcnn_train:16,fasterrcnntrain:16,from:8,gantrain:12,get:6,git:8,handbook:4,hook:[11,12],how:[5,6,8],hyperbol:1,imgop:14,indic:7,inherit:5,init:[11,12],instal:8,issu:5,leaki:1,lifecycl:[11,12],linear:1,linux:6,loop:[11,12],method:[11,12],model:[2,6,10,15],modul:[9,10,11,12,13,14,16,17,18,19,20,21],nativ:[11,12],non:[11,12],organ:5,overview:[5,7],packag:[5,15],pid:6,piecewis:1,pip:8,prerequisit:8,pretrain:2,purpos:5,recent:0,rectifi:1,refer:[15,16,17,18,19,20,21],regist:[13,14],relat:6,relu:1,report:5,repositori:8,sigmoid:1,softplu:1,srgan_train:20,srgantrain:20,step:1,structur:5,tabl:[1,6,7],tangent:1,tanh:1,tensorboard:6,test:[11,12],thi:5,train:[6,11,12],trainer:5,tree:5,tutori:6,unet_train:19,unettrain:19,unit:1,usag:8,use:6,user:6,weight:2,what:5,yolo:6,yolo_train:17,yolotrain:17}})