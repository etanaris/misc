import binascii as ba
from functools import reduce
import sys
import numpy



#Task 1

def hex2_base64(hex_str):
    ''' converts hex string to a base64 srtring'''
    hex_bytes = ba.unhexlify(hex_str)
    print (hex_bytes)
    base64_bytes = ba.b2a_base64(hex_bytes)
    base64_string = base64_bytes.decode(sys.stdout.encoding)
    return base64_string.strip()
    
##------------------TEST---------
##test1 = hex2_base64('49276d206b696c6c696e6720796f757220627261696e206c696b65206120706f69736f6e6f7573206d757368726f6f6d')
##expected1 = 'SSdtIGtpbGxpbmcgeW91ciBicmFpbiBsaWtlIGEgcG9pc29ub3VzIG11c2hyb29t'
##assert test1==expected1, ('you done messed up')

#Task 2

def fixed_XOR(hex_str1, hex_str2):
    ''' Given two buffers (hex strings) returns their XOR combination as a hex string'''
    if len(hex_str1) % 2 != 0:
        hex_str1 = '0'+ hex_str1
    if len(hex_str2) % 2 != 0:
           hex_str2 = '0'+ hex_str2
    hex1_chunk = ba.unhexlify(hex_str1)
    hex2_chunk = ba.unhexlify(hex_str2)
    hex1bytes = [hex1_chunk[i] for i in range(len(hex1_chunk))]
    hex2bytes = [hex2_chunk[i] for i in range(len(hex2_chunk))]
    padding = hex2bytes*abs(len(hex1bytes) - len(hex2bytes))
    
    if len(hex1bytes)> len(hex2bytes):
        temp = padding + hex2bytes
        hex2bytes = temp[0:len(hex1bytes)]
        
    bytewise_xors = [int.to_bytes(xor_val, 1, "big") for xor_val in map(lambda a,b : a^b , hex1bytes, hex2bytes)]
    merged_xor_bytes = reduce(lambda a,b: a+b, bytewise_xors)
##    print (merged_xor_bytes)
    xor_hex = ba.hexlify(merged_xor_bytes)
    xor_hex_str = xor_hex.decode(sys.stdout.encoding).strip() 
    return xor_hex_str, merged_xor_bytes        

##------------------TEST---------------------
##out2 = fixed_XOR('1c0111001f010100061a024b53535009181c', '686974207468652062756c6c277320657965')
##print (out2)
##expected2 = '746865206b696420646f6e277420706c6179'
##assert out2[0] == expected2 , ('you done messed up aaron')

#Task 3
#Etaoin shrdlu
def byte_XOR_cipher(hex_str):
    '''given a hex string that was XOR'd against an unknown single byte, returns the original hex string'''
    byte_hex_strs = [ba.hexlify(int.to_bytes(key, 1, 'big')).decode(sys.stdout.encoding) for key in range(256)]
    possible_msgs= [(fixed_XOR(hex_str, byte_key)[1], byte_key) for byte_key in byte_hex_strs]
    out = {}
    debug = 0
    for msg,byte_key in possible_msgs:
        try:
            out[msg.decode(sys.stdout.encoding)]= byte_key
        except UnicodeDecodeError:
            debug+=1
##            print ('MESSAGE', msg)
            continue
##    print ('DEBUG', debug, len(out))
    original_msg = max(out,key= lambda msg: sum([msg.lower().count(char) for char in 'etaoin shrdlu']))
    return original_msg, out[original_msg]
    

####----------------TEST----------------------
##    
##test3 = byte_XOR_cipher('1b37373331363f78151b7f2b783431333d78397828372d363c78373e783a393b3736')
##print (test3)
##
###Task 4
##
def detect_single_char_XOR(filename):
    '''given a hex_str file, finds the one encrypted by a single byte xor'''
    hex_strs = []
    with open(filename, encoding = sys.stdout.encoding) as file:
        for line in file:
            hex_strs.append(line.strip())
    possible_msgs = {}
    for hex_str in hex_strs:
        msg, key= byte_XOR_cipher(hex_str)
        possible_msgs[msg] = (hex_str, key)
##    print (possible_msgs.keys())
    detected_msg = max(possible_msgs, key= lambda msg: sum([msg.lower().count(char) for char in 'etaoin shrdlu']))
    return detected_msg, possible_msgs[detected_msg]
##
###-------------------TEST--------------
##test4 = detect_single_char_XOR('4.txt')
##print (test4)

###Task 5

def repeating_key_XOR(strs, key_str):
    '''given a list of strings and a key_string, applies a repeating key XOR and returns the resulting hex_string'''
    hex_strs = [ba.hexlify(string.encode(sys.stdout.encoding)) for string in strs]
    key = ba.hexlify(key_str.encode(sys.stdout.encoding))
    return [fixed_XOR(hex_str, key)[0] for hex_str in hex_strs]

###----------------TEST----------------
##inpt = ["Burning 'em, if you ain't quick and nimble\nI go crazy when I hear a cymbal"]
##test5 = repeating_key_XOR(inpt, "ICE")
##expected5 = '0b3637272a2b2e63622c2e69692a23693a2a3c6324202d623d63343c2a26226324272765272\
##a282b2f20430a652e2c652a3124333a653e2b2027630c692b20283165286326302e27282f'
##assert (test5[0] == expected5), 'you done messed up'

###Task 6

def hamming_distance(str1bytes,str2bytes):
    xored = map(lambda b1, b2: b1^b2, str1bytes, str2bytes)
    hamming = sum([bin(val).count('1') for val in xored])
    return hamming

def break_repeating_XOR(filename):

    #opening file
    encrypted_msg_b64 = ''
    with open(filename, encoding= 'cp1252') as file:
        for line in file:
            encrypted_msg_b64+=line.strip()
    encrypted_msg_hex = ba.hexlify(encrypted_msg_b64.encode(sys.stdout.encoding))
    #print(type(encrypted_msg_hex))

    distance = []
    #guessing the optimal key length in bytes
    for guess in range(2,41):
        hd1 = hamming_distance(encrypted_msg_hex[:guess],encrypted_msg_hex[guess:2*guess])
        hd2 = hamming_distance(encrypted_msg_hex[2*guess:3*guess],encrypted_msg_hex[3*guess:4*guess])
        normalized_hd = (hd1+hd2)/(2*guess)
        distance.append((normalized_hd, guess))
        
    distance.sort(key = lambda x: x[0])    
    likely_key_lens = [i[1] for i in distance[1:4]]
    #print (likely_key_lens)

    #constructing blocks to decrypt with the single xor cipher
    possible_keys = []
    for key in likely_key_lens:
        blocks = {}
        count = 0
        for byte in encrypted_msg_hex:
            blk = blocks.get(count % key, b'')
            blk += int.to_bytes(byte, 1, sys.byteorder)
            blocks[count%key] = blk
            count += 1
        #print ('KEY LEN', key, 'BLOCKS', blocks)
            
        possible_key = b''
        for hex_str in list(blocks.values()):
            #print ('HEX_STR', hex_str, len(hex_str))
            possible_key+=ba.unhexlify(byte_XOR_cipher(hex_str.decode(sys.stdout.encoding))[1])
            #print ('POSSIBLE KEY', possible_key)
        
        possible_keys.append(ba.hexlify(possible_key).decode(sys.stdout.encoding))
        
    encrypted_msg_str = encrypted_msg_hex.decode(sys.stdout.encoding)
    possible_msgs= [fixed_XOR(encrypted_msg_str, p_key)[1] for p_key in possible_keys]
    #choosing most likely msg
    likely_msg = max(possible_msgs, key= lambda msg: sum([msg.decode(sys.stdout.encoding).lower().count(char) for char in 'etaoin shrdlu']))
    return likely_msg
##        
##
###-------------------TEST------------------
assert 37==hamming_distance('wokka wokka!!!'.encode('cp1252'), 'this is a test'.encode('cp1252')), 'done messed up'
print (byte_XOR_cipher('894319547c8b441343651431601e46a782e5814ad7a728d28765a1a7d11874f1051815f2838456b293028781940a19c174e57a72c601418ab61465731c1d178e6934346a5d33292848342328a51958c2a7fc1a02a751c036286188075db5f312c4957593858102331415da778db0741d517a1f7620869c227b6449252d416f8715c694321929b43876aa58d8378858a1a367468541b2ac37d84511e45613589794213846545472bad6547117cb84531178731c12099773133443193639772715331b281b4561a728aca7a1152172b731f2cc7198f55923859393158282289413c21205048a8a6812be4152267820c9027581b511a96a313d6317512311113721527b14a212a041682cd5850800a72b461581e59327479484014413044325385115aa26911f5368262f145868d9c272701a447f6f564320452f322063394e00bda8057108277101209b91804717e718e136769b74e40114130820181559915a16a811837f2202197206b52a7024b2ef16652425216111803613b5c4a850a1c2425232209ba1412762a3864821374213943151c202f1539e6b3337459a8a142824b3b2149e7fb598253f53482144105189293046a6f1a92487197da743220540d8074517fa63d7e1346487a753132237e4b89eea946891123c7a2880540381b42173b11572a430922442592325602d46ef912b691128473bac3000c717e817778265422e1d825664c7529678583d6b3348d103b49781ba521a1f628a5a842b2f22a13478a511235d5b3524c5a11679697dd34f4e54a172a55121e741653434a93747e68546477352c53231a697491e317195181a8c13257f7d745218a423b4895153397d1538d9e93e8a1f8189431b2b6162a017f5478621513a542b4246384467a183ae43928511f34877b3d2179512fa047a778e1618a13313ce6f5c8d55012950c44127545dd11c102381671429a11004d105742d71c2333224b8f5475d19f73241030e14f275b7f8b358638a702a1fb7464454aa6c828a81252d189945d25531f5451b5182b919367774a2e76523a3317895b74380a7179287b2473483fdc3a67516d502d71a75934a2274a13719c35c83480a7272c2898d57252df721418b556c2b8621602930a563194b8821825a679b42f812a1e5b4a45a710156154504e7039598a2474a33438a796474a3bf54852316702e2b35c51c5731555833388959251f6071a937f290805e671722165871a557025c52719837011212b6155189174181aa8a796474a907552e206a75111b16a5426721443b5d362965345567786578a233b7b3e8a2111548a0d2210211112629711772925491739135845904448ab4b336150261621475117721727d'))
print (break_repeating_XOR('6.txt'))
##    
##    
##
##    
##    
##    
##                                 
