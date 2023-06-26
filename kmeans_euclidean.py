import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def kmeans_gray_euclidean(image,K,iteration):

    # Burada Gri-Seviyeli Görüntümüz, 1 adet renk kanalı içermektedir ve ona parlaklık(intensity) denmektedir
    # Resim 1 boyutlu sütun vektörüne dönüştürülür yani burada image_vector Nx1'lik bir vektör olacaktır
    image_vector = image.reshape((-1, 1))
    # Küme merkezleri için başlangıç değerlerini rastgele seçiyoruz
    # Burada centers Kx1'lik bir matristir
    np.random.seed(42)
    centers = np.random.randint(0, 256, size=(K, 1))

    # Küme merkezleri için başlangıç değerini kaydediyoruz
    ex_centers = np.copy(centers)


    # K-means algoritmasını uyguluyoruz
    for i in range(iteration):

        # Her piksel için en yakın küme merkezini buluyoruz
        # Görüntümüz, tek kanallı olduğu için uzaklık hesabı farkın mutlak değerine denk gelir yani manhattan distance
        # Burada Nx1'lik vektör olan image_vector ile Kx1'lik matris olan centers arasında gerçekleşen işlemde tür uyuşmazlığını engellemek için:
        # np.newaxis ile centers'ı Kx1x1'lik matrise broadcasting yapıyoruz, böylece her küme merkezi ayrı bir eleman olarak temsil ediliyor ve uyumsuzluk ortadan kalkıyor
        distances = np.abs(image_vector - centers[:, np.newaxis])
        # En küçük uzaklığı veren kümenin index no'sunu alıyoruz, etiket için index değerini kullanıyoruz
        # Örneğin en küçük uzaklık farkı 2.küme içinse o pikseli index no olan 1 ile etiketleriz
        labels = np.argmin(distances, axis=0)
        # Küme merkezlerini güncelleyeceğiz
        for k in range(K):
            # Burada her küme için etiketli pikselleri elde ediyoruz
            cluster_data = image_vector[labels == k]
            # Her küme için o kümenin etiketiyle etiketlenmiş en az, bir piksel var mı kontrolünü sağlıyoruz
            if len(cluster_data) > 0:
                # Küme merkezini ortalamaya göre güncelliyoruz
                new_center = np.mean(cluster_data, axis=0)
                # Eğer yeni küme merkezi bir önceki değeriyle aynı değilse küme merkezini güncelliyoruz
                if not np.array_equal(new_center,centers[k]):
                    centers[k] = new_center


        # Eğer küme merkezlerinin tamamı bir önceki değerleriyle aynıysa algoritmamızı sonlandırıyoruz
        if i > 0:
            difference = np.abs(centers - ex_centers).sum()
            if difference == 0:
                print("Küme merkez noktalari optimal değerlerine {}. döngüde ulaşmiştir...".format(i))
                break

        # Küme merkezlerini bir sonraki güncel değerlerle karşılaştırabilmek için kaydediyoruz
        # Burada aynı zamanda merkezlerin sıfıra çok yakın değerler almasının önüne de geçebiliyoruz
        ex_centers = np.copy(centers)
    

    # Bu kısımda artık elimizde küme merkezleri için optimum değerler mevcuttur
    print("Son küme merkez değerleri: ",centers)

    # Bu değerleri kullanarak resmimizdeki piksellerin atandığı küme değerlerini tutan diziyi elde edebiliriz
    # Yani her piksel için en yakın küme merkezini belirlemiş oluyoruz
    ''' Buradaki dizi şöyle çalışır:
    Her piksel için elde edilen etiketler tek boyutlu bir diziye dönüştürülür ve o etiketin değeri olan index numaraları sayesinde
    Her pikselin en yakın olduğu küme merkez değerini o pikselin bulunduğu konuma atayarak piksel değerlerinin yerine piksellerin en yakın olduğu küme merkez değerlerini tutan diziyi elde ederiz
    '''

    pixel_labels = centers[labels.flatten()]
    final_image = pixel_labels.reshape((image.shape)).astype(np.uint8)


    # Segmentasyon yapılmış görüntümüz için kümelerin histogramlarını gösteren kodu yazıyoruz
    fig, ax = plt.subplots()
    cluster_counts = [np.sum(labels == i) for i in range(K)]
    ax.bar(range(1, K+1), cluster_counts, color=[f'C{i}' for i in range(K)], edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Kümeler')
    ax.set_ylabel('Her Küme İçin Piksel Sayisi')
    ax.set_title('Kümelerin Histogram Grafiği')

    # Küme merkezleri için bir tablo oluşturuyoruz
    centroid_table = pd.DataFrame({'Küme': [f'Küme {i+1}' for i in range(K)],
                                   'Merkez Noktasi': centers.flatten(),
                                   'Piksel Sayisi': cluster_counts})
    centroid_table.set_index('Küme', inplace=True)
    print('\n' + '-'*50)
    print('Gri-Seviyeli Bir Görüntü İçin K-Means Algoritmasi')
    print('-'*50)
    print(centroid_table)

    # Görseli göster
    plt.show()

    # Sonuçları göster
    cv2.imshow('Gri-Seviyeli Goruntu', image)
    cv2.imshow('Kumelendirilmis Goruntu', final_image)
    cv2.waitKey(0)




def kmeans_rgb_euclidean(image,K,iteration):

    # Burada RGB Görüntümüz 3 adet renk kanalı içermektedir, bunlar -> (red,green,blue)
    # Resim 3 boyutlu sütun vektörüne dönüştürülür yani burada image_vector Nx3'lük bir vektör olacaktır
    image_vector = image.reshape((-1, 3))
    # Küme merkezleri için başlangıç değerlerini rastgele seçiyoruz
    # Burada centers Kx3'lük bir matristir
    np.random.seed(42)
    centers = np.random.randint(0, 256, size=(K, 3))


    # Küme merkezleri için başlangıç değerini kaydediyoruz
    ex_centers = np.copy(centers)



     # K-means algoritmasını uyguluyoruz
    for i in range(iteration):


        # Her piksel için en yakın küme merkezini buluyoruz
        # Görüntümüz üç kanallı olduğu için uzaklık hesabı bu sefer klasik öklid uzaklığı formülüyle gerçekleştirilir
        # Burada Nx1'lik vektör olan image_vector ile Kx3'lük matris olan centers arasında gerçekleşen işlemde tür uyuşmazlığını engellemek için:
        # np.newaxis ile centers'ı Kx1x3'lük matrise broadcasting yapıyoruz ve böylece her küme merkezi ayrı bir eleman olarak temsil ediliyor ve uyumsuzluk ortadan kalkıyor
        # axis=2 ile uzaklık hesabının 3.eksen üzerinden (rgb değerlerini tutan eksen) yapılmasını sağlıyoruz
        distances = np.linalg.norm(image_vector - centers[:, np.newaxis], axis=2)
        # En küçük uzaklığı veren kümenin index no'sunu alıyoruz ve etiket için index değerini kullanıyoruz
        # Örneğin en küçük uzaklık farkı 2.küme içinse o pikseli index no olan 1 ile etiketleriz
        labels = np.argmin(distances, axis=0)
        

        # Küme merkezlerini güncelleyeceğiz
        for k in range(K):
            # Burada her küme için etiketli pikselleri elde ediyoruz
            cluster_data = image_vector[labels == k]
            # Her küme için o kümenin etiketiyle etiketlenmiş en az bir piksel var mı kontrolünü sağlıyoruz
            if len(cluster_data) > 0:
                # Küme merkezini ortalamaya göre güncelliyoruz
                new_center = np.mean(cluster_data, axis=0)
                # Eğer yeni küme merkezi bir önceki değeriyle aynı değilse küme merkezini güncelliyoruz
                if not np.array_equal(new_center,centers[k]):
                    centers[k] = new_center


        # Eğer küme merkezlerinin tamamı bir önceki değerleriyle aynıysa algoritmamızı sonlandırıyoruz
        # Burada aynı zamanda merkezlerin sıfıra çok yakın değerler almasının önüne de geçebiliyoruz
        if i > 0:
            difference = np.abs(centers - ex_centers).sum()
            if difference == 0:
                print("Küme merkez noktalari optimal değerlerine {}. döngüde ulaşmiştir...".format(i))
                break

        # Küme merkezlerini bir sonraki güncel değerlerle karşılaştırabilmek için kaydediyoruz
        ex_centers = np.copy(centers)



    # Her piksel için en yakın küme merkezini bul
    print("Son küme merkez değerleri: ",centers)


    # Bu değerleri kullanarak resmimizdeki piksellerin atandığı küme değerlerini tutan diziyi elde edebiliriz
    # Yani her piksel için en yakın küme merkezini belirlemiş oluyoruz
    ''' Buradaki dizi şöyle çalişir:
    Her piksel için elde edilen etiketler tek boyutlu bir diziye dönüştürülür ve o etiketin değeri olan index numaralari sayesinde
    Her pikselin en yakin olduğu küme merkez değerini o pikselin bulunduğu konuma atayarak piksel değerlerinin yerine piksellerin en yakın olduğu küme merkez değerlerini tutan diziyi elde ederiz
    '''
    pixel_labels = centers[labels.flatten()]
    final_image = pixel_labels.reshape((image.shape)).astype(np.uint8)

    # Segmentasyon yapılmış görüntümüz için kümelerin histogramlarını gösteren kodu yazıyoruz
    fig, ax = plt.subplots()
    cluster_counts = [np.sum(labels == i) for i in range(K)]
    #ax.bar(range(1, K+1), cluster_counts, color=[f'C{i}' for i in range(K)], edgecolor='black', linewidth=1.2)
    cluster_colors = [(center[2] / 255, center[1] / 255, center[0] / 255) for center in centers]
    ax.bar(range(1, K+1), cluster_counts, color=cluster_colors, edgecolor='black', linewidth=1.2)
    ax.set_xlabel('Kümeler')
    ax.set_ylabel('Her Küme İçin Piksel Sayisi')
    ax.set_title('Kümelerin Histogram Grafiği')

    # Tablo oluştur
    centroid_table = pd.DataFrame({'Küme': [f'Küme {i+1}' for i in range(K)],
                                   'Merkez Noktasi': [tuple(center) for center in centers],
                                   'Piksel Sayisi': cluster_counts})
    centroid_table.set_index('Küme', inplace=True)
    print('\n' + '-'*50)
    print('RGB Görüntü İçin K-Means Algoritmasi')
    print('-'*50)
    print(centroid_table)

    # Görseli göster
    plt.show()

    # Sonuçları göster
    cv2.imshow('RGB Goruntu', image)
    cv2.imshow('Kumelendirilmis Goruntu', final_image)
    cv2.waitKey(0)
    


img = cv2.imread('scene.jpg')
print("K-Means Algoritmasini RGB Görüntüye Mi Yoksa Gri Seviye Görüntüye Mi Uygulayacağinizi Aşağidaki Bilgilere Göre Seçiniz...")
print("Gri Seviye İçin 1 Girin")
print("RGB İçin 2 Girin")
choose = int(input("Seçiminizi Giriniz: "))
K = int(input("Algoritmada Kullanilacak Dağilim Sayisini Giriniz: "))
iteration = int(input("Algoritmanin Kaç İterasyon Adimi Gerçekleştireceğini Giriniz: "))
if choose == 1:
        cv2.imshow('Kullanilan Goruntu',img)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kmeans_gray_euclidean(img,K,iteration)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
elif choose == 2:
        cv2.imshow('Kullanilan Goruntu',img)
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        kmeans_rgb_euclidean(img,K,iteration)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        exit()
else:
        print("Yanlis bir secim yaptiniz tekrar deneyin...")
exit("Program Sonlandirildi...")

