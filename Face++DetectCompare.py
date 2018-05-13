import requests
import json
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import os
from PIL import Image
def faceDetect(api_key, api_secret, image_url, return_landmark, return_attributes):
    url = 'https://api-cn.faceplusplus.com/facepp/v3/detect'
    param = {
        "api_key": api_key,
        "api_secret": api_secret,
        "image_url": image_url,
        "return_landmark":return_landmark,
        "return_attributes": return_attributes
    }
    data = requests.post(url=url, params=param)
    #r =  json.loads(data.content)
    return data
#传入两张人脸进行比对
def compare(api_key,api_secret,image_url1,image_url2):
    url = 'https://api-cn.faceplusplus.com/facepp/v3/compare'
    param = {
        "api_key":api_key,
        "api_secret":api_secret,
        "image_url1":image_url1,
        "image_url2":image_url2
    }
    response = requests.post(url=url,params=param)
    return response
def main():
    # data = faceDetect("P638X3XA40Aae3kKCyzEtVoGfVwZG2fo", "3HhPQEeqt1VirTIeyQFBdW0jew3sb0o_","http://img4q.duitang.com/uploads/item/201503/21/20150321171207_EX4ae.jpeg", 1,
    #           "gender,age,smiling,glass")
    # print(data.content)
    liyifeng1 = 'http://img3.duitang.com/uploads/item/201501/14/20150114183725_TBMiS.jpeg'
    liyifeng2 = 'http://img4q.duitang.com/uploads/item/201503/21/20150321171207_EX4ae.jpeg'
    liudehua1 ='http://s10.sinaimg.cn/mw690/005MbUK2zy77wjtiGgN19&690'
    liudehua2 ='http://www.qqzhi.com/uploadpic/2014-07-25/092344743.jpg'
    shuzu = [liyifeng1,liyifeng2,liudehua1,liudehua2]
    j=1
    if not os.path.exists('images'):
        os.mkdir('images')
    for item in shuzu:
        #print(item)
        with open('./images/%d.jpg'%j,'wb') as file:
             resp = requests.get(item)
             #print(resp)
             file.write(resp.content)
        lena = mpimg.imread('./images/%d.jpg'%j)
    # 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
        lena.shape  # (512, 512, 3)
        plt.imshow(lena)  # 显示图片
        plt.axis('off')  # 不显示坐标轴
        plt.show()
        j+=1
    data = compare("P638X3XA40Aae3kKCyzEtVoGfVwZG2fo","3HhPQEeqt1VirTIeyQFBdW0jew3sb0o_",liyifeng1,liyifeng2)
    content = json.loads(data.content)
    #print((content))
    result = content['confidence']
    if result>70:
        print("数据比对成功，是同一个人!,相似度为%d"%result)
    else:
        print("相似度为%d,不确定为同一个人！"%result)
if __name__ == '__main__':
    main()