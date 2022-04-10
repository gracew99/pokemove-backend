const pose = {
    keypoints: [
        {
            "x":146.91418552212778,
            "y":102.54829722027101,
            "name":"nose"
        },
        {
            "x":150.16687987329433,
            "y":105.30672688326176,
            "name":"leftEye"
        },
        {
            "x":146.11205150462962,
            "y":103.72193349499916,
            "name":"rightEye"
        }
    ]
}

function draw() {
    var canvas = document.getElementById('canvas');
    if (canvas.getContext) {
      var ctx = canvas.getContext('2d');
  
      ctx.beginPath();
      ctx.moveTo(103.95598877429032, 110.94796333536071);
      ctx.lineTo(133.59815143191094, 137.4365009508635);
      ctx.moveTo(102.1957205908108, 110.87069467169036);
      ctx.lineTo(133.59815143191094, 137.4365009508635);
      ctx.fill();

    }
  }

  draw();

