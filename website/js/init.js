$(document).ready(function(){
    $(".button-collapse").sideNav();
    $('.materialboxed').materialbox();
    $('.scrollspy').scrollSpy();
    $('select').material_select();

}); // end of document ready

function change_task() {

    var x = document.getElementById("task");
    var i = x.selectedIndex;
    var task = x.options[i].value;

    //-- DAVIS
    var video_list = ["bike-packing", "blackswan", "bmx-trees", "breakdance", "camel", "car-roundabout", "car-shadow", "cows", "dance-twirl", "dog", "dogs-jump", "drift-chicane", "drift-straight", "goat", "gold-fish", "horsejump-high", "india", "judo", "kite-surf", "lab-coat", "libby", "loading", "mbike-trick", "motocross-jump", "paragliding-launch", "parkour", "pigs", "scooter-black", "shooting", "soapbox"]

    for (var i=0 ; i < video_list.length ; i++) {

        var video_name = video_list[i];
        var video = document.getElementById(video_name);
        video.src = task + "/DAVIS/" + video_name + ".mp4";
        document.getElementById(video_name).load();    
    }


    //-- videvo
    video_list = ["AircraftTakingOff1", "CoupleRidingMotorbike", "Cycling", "Ducks", "FarmingLandscape", "FatherAndChild2", "Festival", "Freeway2", "Koala", "Madagascar", "MenRidingMotorbike", "PalmTrees", "PoliceCar", "SilverCat", "SkateboarderTableJump", "Surfing", "TimeSquareTraffic", "Vineyard", "Waterfall2", "YogaHut2"]



}