const path = require("path");
const express = require('express');
const bodyParser = require('body-parser');
const pino = require('express-pino-logger')();
const tf = require('@tensorflow/tfjs-node');
const posenet = require('@tensorflow-models/posenet');
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


// read from buffer:
// https://www.npmjs.com/package/canvas
// app.post('/poseNet', (req, res) => {
const poseNet = async() => {
    console.log('start');
    const net = await posenet.load({
        architecture: 'MobileNetV1',
        outputStride: 16,
        inputResolution: 513,
        multiplier: 0.75
      });
      
    const img = new Image();
    img.src = './please.jpg';
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext('2d');
    console.log(img.width, img.height)
    ctx.drawImage(img, 0, 0);
    const input = tf.browser.fromPixels(canvas);
    const pose = await net.estimateSinglePose(input, imageScaleFactor, flipHorizontal, outputStride);
    // console.log(pose);
    for(const keypoint of pose.keypoints) {
        console.log(`${keypoint.part}: (${keypoint.position.x},${keypoint.position.y})`);
    }
    console.log('end');
}
poseNet()
           

//   })


// app.listen(process.env.PORT || 8000, () =>
//     console.log(`Express server is running on localhost:${process.env.PORT || 8000}`)
// );
