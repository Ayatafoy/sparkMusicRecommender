import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class JavaRecommendationExample {

    public static JavaSparkContext sparkContext;
    public static int userID = 30;

    public static void main(String[] args)  {
        long startTime = System.nanoTime();
        SparkConf conf = new SparkConf()
                .setAppName("Java Collaborative Filtering Example")
                .setMaster("local[4]") ;
        sparkContext = new JavaSparkContext(conf);
        String pathToTracks = "/home/cloudera/IdeaProjects/spark/dataSetTracks.csv";
        String pathToArtists = "/home/cloudera/IdeaProjects/spark/dataSetArtists.csv";

        //Read data to RDD (UserID, TrackID, ArtistID, TrackRating)
        List<Integer> listForQualityChecking = new ArrayList();
        Broadcast<List> sharedListForQualityChecking = sparkContext.broadcast(listForQualityChecking);
        JavaRDD<IProduct> trackRatingsRDD = getUserTrackInfoRatingJavaRDD(pathToTracks, sharedListForQualityChecking);
        JavaRDD<IProduct> trackRatingsFilteredRDD = trackRatingsRDD.filter(track -> (!sharedListForQualityChecking.value().contains(track.GetProduct()) || track.GetUser() != userID));

        //Read artists ratings to format (UserID, ArtistID, ArtistRating)
        JavaRDD<IProduct> artistsRatingsRDD = getArtistsRatingJavaRDD(pathToArtists);

        //Fill control user tracks info to shared HashMap (TrackID -> Rating)
        Broadcast<HashMap> sharedControlUserTracksRatingsHashMap = getSharedControlUserProductRatings(trackRatingsFilteredRDD);

        //Fill control user artists info to shared HashMap (ArtistID -> Rating)
        Broadcast<HashMap> sharedControlUserArtistsRatingsHashMap = getSharedControlUserProductRatings(artistsRatingsRDD);

        //Fill similar users tracks to shared HashMap (SimilarUserID -> List<TrackInfo>)
        JavaRDD<IProduct> similarUsersByTracksRDD = getSimilarUsersByTracksRDD(trackRatingsFilteredRDD, sharedControlUserTracksRatingsHashMap);

        Broadcast<HashMap> sharedSimilarUserTracksHashMap = getSharedSimilarUserTracks(similarUsersByTracksRDD);

        //Transformation RDD from (UserID, TrackID, ArtistID, TrackRating) to (UserID, (TrackID, TrackRating))
        JavaPairRDD<Integer, Tuple2> similarUsersTracksRatingsRDD = similarUsersByTracksRDD
                .mapToPair(trackInfo -> new Tuple2(trackInfo.GetUser(), new Tuple2(trackInfo.GetProduct(), trackInfo.GetRating())));

        //Group RDD by userID
        JavaPairRDD<Integer, Iterable<Tuple2>> similarUsersTrackGroupedRDD = similarUsersTracksRatingsRDD.groupByKey();

        //Calculate Rating Of Similarity Users (RatingOfSimilarity, UserID)
        JavaPairRDD<Double, Integer> similarUsersRatingsRDD = similarUsersTrackGroupedRDD
                .mapToPair(userTrackRating -> calculateRatingOfSimilarityUsers(sharedControlUserTracksRatingsHashMap, userTrackRating));


        //Sort users by rating of similarity
        List<Tuple2<Double, Integer>> similarFirstHundreadUsersSortedList = similarUsersRatingsRDD.sortByKey(false).take(100);

        JavaPairRDD<Double, Integer> similarFirstHundreadUsersSortedRDD = sparkContext.parallelizePairs(similarFirstHundreadUsersSortedList);
        //Fill similar users all tracks list to shared HashMap (SimilarUserID -> List<TrackInfo>)
        JavaRDD<IProduct> similarUserAllTrackEvalRDD = trackRatingsFilteredRDD
                .filter(userTrackInfoRating -> sharedSimilarUserTracksHashMap.value().containsKey(userTrackInfoRating.GetUser()));
        Broadcast<HashMap> sharedSimilarUserAllTracksRatingsHashMap = getSharedSimilarUserTracks(similarUserAllTrackEvalRDD);


        //For each user get sorted users tracks by rating of evaluation this track by control user
        List<List<UserCustomProductRating>> similarUserTracksList = similarFirstHundreadUsersSortedRDD
                .map(item ->
                        getSortedUserTracks(sharedControlUserArtistsRatingsHashMap, sharedSimilarUserAllTracksRatingsHashMap, item._2)).collect();

        //Prepare and out recommended tracks
        List<Integer> resultList = getResultList(similarUserTracksList);
        AtomicInteger quality = new AtomicInteger();

        resultList.forEach(trackID -> {
            System.out.println(trackID);
            if (sharedListForQualityChecking.value().contains(trackID))
                quality.getAndIncrement();
        });
        long endTime = System.nanoTime();
        System.out.println("Took "+ (endTime - startTime) / 1000000000 + " seconds;");
        System.out.println("Quality of recommendations: " + quality);
    }

    private static List<Integer> getResultList(List<List<UserCustomProductRating>> sortedList) {
        //list(0).sublist(0), list(0).sublist(1), list(1).sublist(0), list(0).sublist(2), list(1).sublist(1), list(2).sublist(0)
        List resultList = new ArrayList();
        int i = 0;
        while (i < sortedList.size()) {
            int j = 0;
            while (j <= i) {
                if (i - j < sortedList.get(j).size() && j < sortedList.size()) {
                    int productID = sortedList.get(j).get(i - j).GetProduct();
                    if (!resultList.contains(productID))
                        resultList.add(productID);
                    if (resultList.size() == 50)
                        break;
                }
                j++;
            }
            if (resultList.size() == 50)
                break;
            i++;
        }
        return resultList;
    }

    private static JavaRDD<IProduct> getArtistsRatingJavaRDD(String pathArtists) {
        JavaRDD<String> input = sparkContext.textFile(pathArtists);
        return input.map(inputLine -> {
            String[] array = inputLine.split(",");
            return new UserProductRating(Integer.parseInt(array[0]),
                    Integer.parseInt(array[1]),
                    Double.parseDouble(array[2]));
        });
    }

    private static JavaRDD<IProduct> getUserTrackInfoRatingJavaRDD(String pathTracks, Broadcast<List> sharedListForQualityChecking) {
        JavaRDD<String> input = sparkContext.textFile(pathTracks);
        final Integer[] counter = {1};
        return input
                .map(inputLine -> {
                    String[] array = inputLine.split(",");
                    if (Integer.parseInt(array[0]) == userID && counter[0] % 2 == 0)
                        sharedListForQualityChecking.value().add(Integer.parseInt(array[1]));
                    counter[0]++;
                    return new UserCustomProductRating(
                            Integer.parseInt(array[0]),
                            Integer.parseInt(array[1]),
                            Integer.parseInt(array[2]),
                            Double.parseDouble(array[3]));
                });
    }

    private static Broadcast<HashMap> getSharedSimilarUserTracks(JavaRDD<IProduct> similarUsersByTracksRDD) {
        List<IProduct> similarUsersByTracksList = similarUsersByTracksRDD.collect();
        HashMap<Integer, List<IProduct>> similarUserHashMap = new HashMap<>();
        similarUsersByTracksList.forEach(similarUserTrackInfo -> {
            if (similarUserHashMap.get(similarUserTrackInfo.GetUser()) == null){
                similarUserHashMap.put(similarUserTrackInfo.GetUser(), new ArrayList<>());
            }
            similarUserHashMap.get(similarUserTrackInfo.GetUser()).add(similarUserTrackInfo);
        });
        return sparkContext.broadcast(similarUserHashMap);
    }

    private static Broadcast<HashMap> getSharedControlUserProductRatings(JavaRDD<IProduct> productRatingsRDD) {
        List<IProduct> controlUserProductRatingsList = productRatingsRDD
                .filter(userProductInfo -> (userProductInfo.GetUser() == userID))
                .collect();
        HashMap<Integer, Double> controlUserProductRatingsHashMap = new HashMap<>();
        controlUserProductRatingsList.forEach(userProductInfo -> controlUserProductRatingsHashMap.put(userProductInfo.GetProduct(), userProductInfo.GetRating()));
        return sparkContext.broadcast(controlUserProductRatingsHashMap);
    }

    private static JavaRDD<IProduct> getSimilarUsersByTracksRDD(JavaRDD<IProduct> trackRatingsRDD, Broadcast<HashMap> sharedControlUserTracksRatingsHashMap) {
        return trackRatingsRDD
                    .filter(userTrackInfo -> (sharedControlUserTracksRatingsHashMap
                            .value()
                            .containsKey(userTrackInfo.GetProduct()) && userTrackInfo.GetUser() != userID));
    }


    private static List<UserCustomProductRating> getSortedUserTracks(
            Broadcast<HashMap> sharedControlUserArtistsRatingsHashMap,
            Broadcast<HashMap> sharedSimilarUserAllTracksRatingsHashMap,
            Integer userId) {

        //Read user List<TrackInfo> from shared hashMap
        ArrayList<UserCustomProductRating> friendTracksInfoList = (ArrayList<UserCustomProductRating>)
                sharedSimilarUserAllTracksRatingsHashMap
                        .value()
                        .get(userId);

        //For each tracks calculate control user rating of this track and norm this rating by natural logarithm
        friendTracksInfoList.forEach(userTrackInfo -> {
            userTrackInfo.SetRating(0.0);
            if (sharedControlUserArtistsRatingsHashMap.value().containsKey(userTrackInfo.GetProductAttribute())){
                userTrackInfo.SetRating((double) sharedControlUserArtistsRatingsHashMap.value().get(userTrackInfo.GetProductAttribute()));
            }
        });

        //Sort tracks by rating
        friendTracksInfoList.sort(Collections.reverseOrder(Comparator.comparing(userTrackInfo -> userTrackInfo.GetRating())));
        return friendTracksInfoList;
    }

    private static Tuple2<Double, Integer> calculateRatingOfSimilarityUsers(
            Broadcast<HashMap> sharedControlUserTracksEvalsHashMap,
            Tuple2<Integer, Iterable<Tuple2>> userTracks) {

        //Calculate rating of similarity users with control user
        Iterator tracks = userTracks._2.iterator();
        double resultEval = 0.0;
        while (tracks.hasNext()){
            Tuple2 productEval = (Tuple2) tracks.next();
            double friendUserEval = (double) productEval._2;
            double controlUserEval = (double) sharedControlUserTracksEvalsHashMap.value().get(productEval._1);
            resultEval += friendUserEval + controlUserEval - Math.abs(friendUserEval - controlUserEval);
        }
        return new Tuple2(resultEval, userTracks._1);
    }
}