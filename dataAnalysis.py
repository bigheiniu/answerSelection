import os
import re
def max_value_cal(line_list):
    line_index = 0
    acc = 0.
    mAP = 0.
    mRP = 0.
    f1 = 0.
    Pat1 = 0.
    count = 0.
    fuck_len = len(line_list)
    while line_index < len(line_list):
        if "Key: lstm_hidden_size" in line_list[line_index]:
            line_index += 1
            max_Pat1 = 0.
            max_acc = 0.
            max_mAP = 0.
            max_mRP = 0.
            max_f1 = 0.
            while line_index < len(line_list) and "Key: lstm_hidden_size" not in line_list[line_index]:
                line = line_list[line_index]
                if "[Info]" in line:
                    # value = re.search( "p@1: ([0-9/.]+), ACC: ([0-9/.]+), mAP: ([0-9/.]+), MRP: ([0-9/.]+), f1: ([0-9/.]+),", line)
                    #[Info] mAP: 0.7337601846992527, P@1: 0.6893052302888368, mRP: 0.7690525754928571, Accuracy: 0.6253753753753754, loss: 0.010360080388573197, one_count: 3626, zero_count: 3034
                    # Pat1_ = float((re.search("P@1: ([0-9/.]+),", line)).group(1))
                    Pat1_ = float((re.search("p@1: ([0-9/.]+),", line)).group(1))
                    acc_ = float((re.search("ACC: ([0-9/.]+),", line)).group(1))
                    # acc_ = float((re.search("Accuracy: ([0-9/.]+),", line)).group(1))
                    mAP_ = float((re.search("mAP: ([0-9/.]+),",line).group(1)))
                    mRP_ = float((re.search("MRP: ([0-9/.]+),",line).group(1)))
                    f1_ = float((re.search("f1: ([0-9/.]+),", line).group(1)))
                    # acc_, mAP_, mRP_, f1_ = map(lambda x: float(x), value.groups())
                    if Pat1_ > max_Pat1:
                        max_Pat1 = Pat1_
                    if acc_ > max_acc:
                        max_acc = acc_
                    if mAP_ > max_mAP:
                        max_mAP = mAP_
                    if mRP_ > max_mRP:
                        max_mRP = mRP_
                    if f1_ > max_f1:
                        max_f1 = f1_

                line_index += 1

            acc += max_acc
            mAP += max_mAP
            mRP += max_mRP
            f1 += max_f1
            Pat1 += max_Pat1
            count += 1
        else:
            line_index += 1
    return Pat1 / count, acc / count, mAP / count, mRP / count, f1 /count, count

def read_file(filename):
    with open(filename, 'r') as f1:
        line_list = f1.readlines()
    value = max_value_cal(line_list)
    print("[INFO] p@1: {}, avg acc: {}, avg mAP: {}, avg mRP: {}, avg f1: {}, count: {}".format(*value))

if __name__ == '__main__':
    fileName = "cntn_graph.log"
    read_file(fileName)


