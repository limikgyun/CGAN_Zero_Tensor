from scapy.all import rdpcap, wrpcap

def merge_pcap_files(pcap_files, output_file):
    # 패킷을 저장할 리스트 초기화
    all_packets = []

    # 각 PCAP 파일에서 패킷을 읽어와서 리스트에 추가
    for pcap_file in pcap_files:
        packets = rdpcap(pcap_file)
        all_packets.extend(packets)

    # 모든 패킷을 하나의 PCAP 파일로 저장
    wrpcap(output_file, all_packets)
    print(f'Merged {len(pcap_files)} PCAP files into {output_file}')

# 첫 번째 그룹: c_11~c_15 -> c_1.pcap
pcap_files_1 = [
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_11.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_12.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_13.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_14.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_15.pcap'
]
output_file_1 = '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_1.pcap'
merge_pcap_files(pcap_files_1, output_file_1)

# 두 번째 그룹: c_21~c_25 -> c_2.pcap
pcap_files_2 = [
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_21.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_22.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_23.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_24.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_25.pcap'
]
output_file_2 = '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_2.pcap'
merge_pcap_files(pcap_files_2, output_file_2)

# 세 번째 그룹: c_31~c_35 -> c_3.pcap
pcap_files_3 = [
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_31.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_32.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_33.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_34.pcap',
    '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_35.pcap'
]
output_file_3 = '/home/mnetlig/Desktop/CSI-SemiGAN-master/dataset/c_3.pcap'
merge_pcap_files(pcap_files_3, output_file_3)