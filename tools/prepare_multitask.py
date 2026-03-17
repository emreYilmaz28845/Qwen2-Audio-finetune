import argparse
import re
import json

class prepare_multitask:
    def __init__(self):
        self.data = []
    def append_data(self,text,task_name="ASR"):
        with open(text) as f:
            for line in f:
                item = {}
                key,text = line.replace("\t"," ").split(" ",1)
                text = text.strip().replace(" ","")
                if text != "":
                    item["key"] = key
                    item["task"] = task_name
                    item["target"] = text.strip().lower().replace('"',"")
                    self.data.append(item)
                    # print(self.data[-1])
    def append_data_info(self,text,info,info_name):
        with open(text) as f1:
            with open(info) as f2:
                for item1,item2 in zip(f1,f2):
                    item = {}
                    key1,text1 = item1.replace("\t"," ").strip("\n").split(" ",1)
                    key2,text2 = item2.replace("\t"," ").strip("\n").split(" ",1)
                    assert(key1 == key2)
                    if text2 != "":
                        item["key"] = key1
                        item["task"] = info_name
                        item["target"] = text1.strip().lower().replace('"',"")
                        item[info_name] = text2.strip().lower().replace('"',"")
                        self.data.append(item)
                    else:
                        item["key"] = key1
                        item["task"] = "ASR"
                        item["target"] = text1.strip().lower().replace('"',"")
                        self.data.append(item)
                        pass
                        # print(self.data[-1])
    def save(self,out_file):
        with open(out_file,"w") as f:
            for item in self.data:
                f.write(json.dumps(item,ensure_ascii=False)+"\n")


if __name__ == "__main__":
    pass
    # train
    # asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/train/text")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/train/text")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/train/text")

    # ## mt
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/mt/train_mt","ZH2EN")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/train_mt","EN2ZH")

    ## hotword
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/train/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/hotword/train_hotword","hotword")
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/train/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/hotword/train_hotword","hotword")
    
    
    # ## prevtext
    ### 1
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/train/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/train_1","prevtext")
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/train/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/train_3","prevtext")
    
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/multitask/train/multitask.jsonl")

    # # # dev
    # # ## asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/dev/text")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/dev/iOS/text")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-clean/text")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-other/text")

    # ## mt
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/mt/dev_mt","ZH2EN")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/dev-clean_mt","EN2ZH")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/dev-other_mt","EN2ZH")
    
    # ## hotword
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/dev/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/hotword/dev_hotword","hotword")
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/dev/iOS/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/hotword/dev_hotword","hotword")
    
    
    # ## prevtext
    # ### 1
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-clean/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/dev-clean_1","prevtext")
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-clean/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/dev-clean_3","prevtext")    
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-other/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/dev-other_1","prevtext")    
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-other/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/dev-other_3","prevtext")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/multitask/dev/multitask.jsonl")

    # # # test
    # # ## asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/test/text")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/test/iOS/text")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-clean/text")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-other/text")

    # # ## mt
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/mt/test_mt","ZH2EN")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/test-clean_mt","EN2ZH")
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/test-other_mt","EN2ZH")
    
    # ## hotword
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/test/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/hotword/test_hotword","hotword")
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/test/iOS/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/hotword/test_hotword","hotword")
    
    
    # # ## prevtext
    # # ### 1
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-clean/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/test-clean_1","prevtext")
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-clean/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/test-clean_3","prevtext")
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-other/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/test-other_1","prevtext")
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-other/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/test-other_3","prevtext")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/multitask/test/multitask.jsonl")


    ################################################################################

    # aishell-1
    # # train
    # ## asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/train/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/train/multitask.jsonl")
    # ## hotword
    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/train/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/hotword/train_hotword","hotword")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/hotword/train/multitask.jsonl")
    # # dev
    # ## asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/dev/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/dev/multitask.jsonl")
    # ## hotword
    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/dev/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/hotword/dev_hotword","hotword")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/hotword/dev/multitask.jsonl")
    # # test
    # ## asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/test/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/test/multitask.jsonl")
    # # hotword
    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/asr/test/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/hotword/test_hotword","hotword")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-1/hotword/test/multitask.jsonl")

    ###########################################################################################

    # aishell-2
   # # train
    # ## asr

    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/train/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/train/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/mt/train_mt","ZH2EN")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/mt/train/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/train/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/hotword/train_hotword","hotword")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/hotword/train/multitask.jsonl")

    # # dev
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/dev/iOS/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/dev/iOS/multitask.jsonl")


    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/mt/dev_mt","ZH2EN")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/mt/dev/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/dev/iOS/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/hotword/dev_hotword","hotword")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/hotword/dev/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/test/iOS/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/test/multitask.jsonl")

    # test
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/mt/test_mt","ZH2EN")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/mt/test/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/asr/test/iOS/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/hotword/test_hotword","hotword")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/aishell-2/hotword/test/multitask.jsonl")

    #############################################################################################

    # librispeech
    # # train
    # ## asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/train/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/train/multitask.jsonl")

    ## mt
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/train_mt","EN2ZH")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/train/multitask.jsonl")

    
    # ## prevtext
    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/train/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/train_1","prevtext")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/train/multitask.jsonl")

    # # dev
    # ## asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-clean/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-clean/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-other/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-other/multitask.jsonl")



    # ## mt
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/dev-clean_mt", "EN2ZH")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/dev-clean/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/dev-other_mt", "EN2ZH")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/dev-other/multitask.jsonl")

    # ## hotword
    
    
    ## prevtext
    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-clean/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/dev-clean_1","prevtext")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/dev-clean/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/dev-other/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/dev-other_1","prevtext")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/dev-other/multitask.jsonl")
    # ### 1


    # # test
    # ## asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-clean/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-clean/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-other/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-other/multitask.jsonl")

    ## mt
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/test-clean_mt","EN2ZH")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/test-clean/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/test-other_mt","EN2ZH")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/mt/test-other/multitask.jsonl")

    # # hotword
    
    
    ## prevtext
    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-clean/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/test-clean_1","prevtext")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/test-clean/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/asr/test-other/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/test-other_1","prevtext")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/librispeech/prevtext/test-other/multitask.jsonl")


    ########################################################

    # wenet 

    # # test
    # ## asr
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/wenet/test_meeting/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/wenet/test_meeting/multitask.jsonl")

    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/wenet/test_net/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/wenet/test_net/multitask.jsonl")


    ################################################################
    # 线上数据集
    # prep = prepare_multitask()
    # prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/test/trans_task_test_0125/text")
    # prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/test/trans_task_test_0125/multitask.jsonl")


    #####################################################################
    # alimeeting
    # train
    ## asr
    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf/train/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf/train/multitask.jsonl")

    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_gss/train/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_gss/train/multitask.jsonl")

    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf_sot/train/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf_sot/train/multitask.jsonl")

    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_near/train/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_near/train/multitask.jsonl")

    # dev
    ## asr
    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf/dev/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf/dev/multitask.jsonl")

    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_gss/dev/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_gss/dev/multitask.jsonl")

    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf_sot/dev/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf_sot/dev/multitask.jsonl")

    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_near/dev/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_near/dev/multitask.jsonl")

    # test
    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf/test/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf/test/multitask.jsonl")

    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_gss/test/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_gss/test/multitask.jsonl")

    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf_sot/test/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_far_bf_sot/test/multitask.jsonl")

    prep = prepare_multitask()
    prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_near/test/text")
    prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/alimeeting/asr_near/test/multitask.jsonl")

    ###################################################

#     # slidespeech
#     ## asr
#     prep = prepare_multitask()
#     prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/train/text")
#     prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/train/multitask.jsonl")
#     prep = prepare_multitask()
#     prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/dev/text")
#     prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/dev/multitask.jsonl")
#     prep = prepare_multitask()
#     prep.append_data("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/test/text")
#     prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/test/multitask.jsonl")

#    ## hotword
#     prep = prepare_multitask()
#     prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/train/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/hotword/L95_hotword","hotword")
#     prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/hotword/train/multitask.jsonl")
#     prep = prepare_multitask()
#     prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/dev/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/hotword/dev_hotword","hotword")
#     prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/hotword/dev/multitask.jsonl")
#     prep = prepare_multitask()
#     prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/test/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/hotword/test_howtord","hotword")
#     prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/hotword/test/multitask.jsonl")

#     ## domain
#     prep = prepare_multitask()
#     prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/train/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/domain/train_domain","domain")
#     prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/domain/train/multitask.jsonl")
#     prep = prepare_multitask()
#     prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/dev/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/domain/dev_domain","domain")
#     prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/domain/dev/multitask.jsonl")
#     prep = prepare_multitask()
#     prep.append_data_info("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/asr/test/text","/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/domain/test_domain","domain")
#     prep.save("/hpc_stor01/home/yangui.fang_sx/workingspace/data/slidespeech/domain/test/multitask.jsonl")