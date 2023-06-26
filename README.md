# K-Means-Algorithm-Euclidean-Distance

Türkçe:

Öklid Uzaklığını Kullanarak K-Means Algoritmasıyla Bir Görüntüde Segmentasyon İşlemi

NOT: Bu algoritmada hem gri-seviyeli görüntü için hem de rgb görüntü için 2 ayrı fonksiyon kodlanmıştır. 
NOT 2: Kodumuzda resmi yükleme şeklini kendinize göre değiştirmeniz gerekmektedir.

K-Means Algoritması kullanarak bir görüntüde segmentasyon yapıyoruz.
Öncelikle bu algoritmada öklid uzaklığının kullanılması, kümelerdeki her noktayı dikkate almak yerine bu kümelerin dairesel olarak modellendiğini varsayıp ağırlık merkezlerini dikkate almak demek oluyor.
Böylece her pikselin, her bir kümenin ağırlık merkezine(center or centroid) olan öklid uzaklığını hesaplayarak minimum uzaklığı veren kümeyi ararız.
Küme merkezlerinin tamamının değişim göstermediği durumda ise optimal merkez noktalarını bulduk demektir.
Sonuç olarak optimal küme merkez noktalarına göre bütün pikselleri etiketler ve segmentasyon sonucu oluşan görüntümüzü elde ederiz.

English:

K-Means Algorithm For Image Segmentation Using Euclidean Distance

NOTE: Two separate functions have been coded for both grayscale and RGB images in this algorithm.
NOTE 2: You need to modify the image loading part according to your own requirements in the code.

We are performing image segmentation using the K-Means algorithm, incorporating the use of Euclidean distance.
In this algorithm, utilizing Euclidean distance means assuming that the clusters are modeled as circular and considering the centroids of these clusters instead of individually considering each point in the clusters.
By calculating the Euclidean distance between each pixel and the centroid of each cluster, we search for the cluster that provides the minimum distance.
If the positions of all cluster centers remain unchanged, it indicates that we have found the optimal center points.
Consequently, we label all pixels according to the optimal cluster center points and obtain the resulting image from the segmentation process.
