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

const ref1 = { // warrior
    nose: [ 248.6164775647615, 114.84564530222036 ],
    leftEye: [ 252.03414113898026, 112.41429379111838 ],
    rightEye: [ 243.84267706620065, 110.14147306743422 ],
    leftEar: [ 262.30492842824833, 116.67213841488484 ],
    rightEar: [ 236.85328433388156, 114.5601613898026 ],
    leftShoulder: [ 271.73683568050984, 135.2590139288651 ],
    rightShoulder: [ 226.55058208264802, 134.25707365337172 ],
    leftElbow: [ 311.84101305509864, 150.80315840871708 ],
    rightElbow: [ 167.6443963301809, 140.58217901932562 ],
    leftWrist: [ 354.79774876644734, 150.78054327713812 ],
    rightWrist: [ 139.6556332236842, 138.16891318873354 ],
    leftHip: [ 268.53032162314963, 216.13377621299338 ],
    rightHip: [ 239.92367393092104, 217.6985891241776 ],
    leftKnee: [ 298.6618523848684, 255.42464406866776 ],
    rightKnee: [ 199.9729517886513, 270.367431640625 ],
    leftAnkle: [ 332.12521201685854, 314.9677477384868 ],
    rightAnkle: [ 212.757568359375, 323.7043842516447 ]
}

const ref2 = { // tree
    nose: [ 250.15000192742596, 89.41730700041114 ],
    leftEye: [ 252.31967323704768, 86.8032033819901 ],
    rightEye: [ 244.26035027754932, 86.16284822162828 ],
    leftEar: [ 261.95182398745885, 90.11355751439146 ],
    rightEar: [ 231.4698871813322, 90.12335526315786 ],
    leftShoulder: [ 273.21314761513156, 101.19053890830588 ],
    rightShoulder: [ 221.8167596114309, 106.36892218338812 ],
    leftElbow: [ 307.2571122018914, 80.53160014905427 ],
    rightElbow: [ 195.40076004831414, 77.50191136410359 ],
    leftWrist: [ 317.68904836554276, 53.40201929995888 ],
    rightWrist: [ 197.27672376130755, 53.05376554790294 ],
    leftHip: [ 260.02513684724505, 183.1223658511513 ],
    rightHip: [ 231.10166850842927, 178.6354787726151 ],
    leftKnee: [ 259.74056846217104, 233.14504523026312 ],
    rightKnee: [ 220.01961155941612, 224.58807694284536 ],
    leftAnkle: [ 261.74435264185854, 267.3027600740131 ],
    rightAnkle: [ 228.02462929173518, 257.7023797286184 ]
}
const ref3 = {     // difficult
    nose: [ 228.12885485197367, 76.86785246196547 ],
    leftEye: [ 230.24174740439966, 71.19359066611841 ],
    rightEye: [ 225.32920435855263, 72.94176603618419 ],
    leftEar: [ 237.18936317845393, 70.08567408511513 ],
    rightEar: [ 208.45091167249177, 73.13577752364307 ],
    leftShoulder: [ 246.6487041272615, 92.48644377055922 ],
    rightShoulder: [ 205.44015984786182, 98.26859323601974 ],
    leftElbow: [ 290.0695158305921, 100.13218929893088 ],
    rightElbow: [ 150.9707256116365, 105.90913471422698 ],
    leftWrist: [ 386.03107653166114, 114.60924650493422 ],
    rightWrist: [ 89.1465357730263, 131.3029078433388 ],
    leftHip: [ 214.18119731702302, 174.95679353412828 ],
    rightHip: [ 202.45732357627466, 170.70929276315786 ],
    leftKnee: [ 281.3408138877467, 152.54715768914474 ],
    rightKnee: [ 227.8098016036184, 246.51344700863484 ],
    leftAnkle: [ 389.64800382915294, 141.36725174753286 ],
    rightAnkle: [ 211.35710063733552, 301.7229260896381 ]
}

const refs = [ref3, ref2, ref1]

function isSimilar(pose, ref, width, height) {
    let mistake = 0;
    for(const keypoint of pose.keypoints) {
        const part = keypoint.part;
        const x = keypoint.position.x;
        const y = keypoint.position.y;
        console.log("x", ref[part][0], x, Math.abs(ref[part][0] - x), Math.abs(ref[part][0] - x)/width)
        console.log("y", ref[part][1], y, Math.abs(ref[part][1] - y), Math.abs(ref[part][1] - y)/height)
        if ((Math.abs(ref[part][0] - x) /width) >= 0.05) {
            mistake += 1;
        }
        if ((Math.abs(ref[part][1] - y)/height) >= 0.05) {
            mistake += 1;
        }
        if (mistake >= 3) {
            
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
        const result =  isSimilar(pose, refs[req.body.refIdx], 1080, 720);
        console.log("refidx", req.body.refIdx)
        const received = {}
        for(const keypoint of pose.keypoints) {
            received[keypoint.part] = [keypoint.position.x, keypoint.position.y]
            // console.log(`${keypoint.part}: (${keypoint.position.x},${keypoint.position.y})`);
        }
        // console.log(received)
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
