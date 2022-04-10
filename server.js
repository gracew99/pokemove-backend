const path = require("path");
const express = require('express');
const bodyParser = require('body-parser');
const pino = require('express-pino-logger')();
const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
const fs = require("fs");

const {
    createCanvas, Image
} = require('canvas')
const imageScaleFactor = 0.5;
const outputStride = 16;
const flipHorizontal = false;

const app = express();
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.use(pino);
app.use(express.static(path.join(__dirname, "..", "build")));

app.use(express.json());
app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*"),
  res.setHeader("Access-Control-Allow-Headers", "*"),
  next();
})

const ref1 = {
    // nose: [592.2518117804276,106.68036209909536],
    // leftEye: [603.3253880550986,98.55577970805922],
    // rightEye: [583.311767578125,93.36946186266448],
    // leftEar: [619.0220240542762,112.74654990748354],
    // rightEar: [564.8036595394736,105.50983629728614],
    // leftShoulder: [634.2262027138157,169.76495040090458],
    // rightShoulder: [542.7144582648026,150.40720086348682],
    // leftElbow: [713.6811266447368,127.99698679070724],
    // rightElbow: [473.63339072779604,103.99526495682562],
    // leftWrist: [530.6711939761512,40.94810084292763],
    // rightWrist: [519.1156005859375,41.64120322779604],
    // leftHip: [594.9927400287828,327.7620014391447],
    // rightHip: [517.7156147203947,307.7296528063322],
    // leftKnee: [580.1476408305921,478.2258686266447],
    // rightKnee: [452.60151110197364,396.0979260896381],
    // leftAnkle: [548.4050549958881,632.8867701480262],
    // rightAnkle: [499.6025968852796,543.8245271381578],
    nose: [227.9814388877467,73.74709279913651],
leftEye: [231.0845947265625,70.11050575657893],
rightEye: [217.5311439915707,70.00714753803453],
leftEar: [230.17957185444078,76.934814453125],
rightEar: [212.77273077713815,73.89752839740953],
leftShoulder: [248.33179674650492,94.730224609375],
rightShoulder: [205.14571340460526,88.71720163445724],
leftElbow: [278.0068166632401,90.86464329769734],
rightElbow: [183.59400699013156,62.43310225637333],
leftWrist: [270.02377158717104,61.47099545127466],
rightWrist: [197.8896131013569,55.84887052837169],
leftHip: [230.36935906661182,172.4823319284539],
rightHip: [202.55796733655427,166.99369731702302],
leftKnee: [221.64027163856906,214.91712068256578],
rightKnee: [196.99912623355263,208.38292172080588],
leftAnkle: [223.8008037366365,240.52882144325656],
rightAnkle: [204.40121299342104,232.5291041324013],
}

function isSimilar(pose, ref, width, height) {
    let mistake = 0;
    for(const keypoint of pose.keypoints) {
        const part = keypoint.part;
        const x = keypoint.position.x;
        const y = keypoint.position.y;
        console.log("x", ref[part][0], x, Math.abs(ref[part][0] - x), Math.abs(ref[part][0] - x)/width)
        console.log("y", ref[part][1], y, Math.abs(ref[part][1] - y), Math.abs(ref[part][1] - y)/height)
        if ((Math.abs(ref[part][0] - x) /width) >= 0.11) {
            mistake += 1;
        }
        if ((Math.abs(ref[part][1] - y)/height) >= 0.11) {
            mistake += 1;
        }
        if (mistake >= 4) {
            return false
        }
    }
    console.log("mistake: ", mistake) 
    return mistake < 4
}
// read from buffer:
// https://www.npmjs.com/package/canvas
app.post('/poseNet', (req, res) => {
    // console.log(req.body.imgSrc)
    const poseNet = async() => {
        console.log('start');
        const net = await posenet.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            inputResolution: 513,
            multiplier: 0.75
        });
        // const buffer = Buffer.from(req.body.imgSrc, "base64");
        // console.log(req.body.imgSrc)
        // fs.writeFileSync("new-path.jpg", buffer);

        var image = new Image();
        

        const canvas = createCanvas(1080, 720);
        const ctx = canvas.getContext('2d');
        // console.log(img.width, img.height)
        // ctx.drawImage(img, 0, 0);

        image.onload = function() {
        ctx.drawImage(image, 0, 0);
        };
        image.src = req.body.imgSrc


        const input = tf.browser.fromPixels(canvas);
        const pose = await net.estimateSinglePose(input, imageScaleFactor, flipHorizontal, outputStride);
        const result =  isSimilar(pose, ref1, 1080, 720);
        const received = {}
        for(const keypoint of pose.keypoints) {
            received[keypoint.part] = [keypoint.position.x, keypoint.position.y]
            console.log(`${keypoint.part}: (${keypoint.position.x},${keypoint.position.y})`);
        }
        console.log(received)
        return {
            isMatch: result,
            skeleton: received
        };
    }
    poseNet().then(isMatch => {console.log(isMatch); res.send(isMatch)})
    // console.log(isMatch)
    // res.send(isMatch)
  })


app.listen(8001, () =>
    console.log(`Express server is running on localhost:${8001}`)
);
