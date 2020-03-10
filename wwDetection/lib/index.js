var specGenerator;
var model; 
let threshold = 0.99;
let count = 0;
var chartData = [];
var xVal = 0;
var fpMode = false;
var modelName;
var date = new Date();
const NUM_FRAMES = numFramesPerSpectrogramValue;
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let  TIMESTEPS = 0;
window.onload = init;

function init(){
  async function app() {
    specGenerator = speechCommands.create();
    document.getElementById('fp_mode').innerHTML = "False Positive mode:<br />" + fpMode;
  }
  
  app().then((result)=>{
    document.getElementById('listen').disabled = false;
    document.getElementById('predictions').style.display = 'flex';
  });
}

async function loadModel(modelName, timeSteps) {
  document.getElementById('modelDropdown').innerHTML = modelName;
  TIMESTEPS = timeSteps;
  model = await tf.loadLayersModel(inputUrl+'/models/'+ modelName +'/model.json');
  console.log(modelName);
  updateChart(0);
  listen();
  setTimeout(listen, 5000);
}

function highlight(confidence) {
  updateChart(confidence);
  pred_divs = document.getElementsByClassName('prediction');
  for(let i=0;i<pred_divs.length;i++){
    pred_divs[i].classList.remove('green_background');
  }
  if((confidence >confidenceThreshold)){
    document.getElementById('ok_atlas').innerHTML = 'ok_Atlas<br>' + confidence.toFixed(5);
    document.getElementById('ok_atlas').classList.add('green_background');
    console.log("OK ATLAS DETECTED!")
  }
  else {
    let val = 1.000 - confidence.toFixed(5);
    document.getElementById('other').innerHTML = 'other<br>' + val;
    document.getElementById('other').classList.add('red_background');
  }
}

function setThreshold(){
  let value = document.getElementById("threshold").value;
  console.log("update threshold: ",value);
  confidenceThreshold = value;
}

function download(content, fileName, contentType) {
    content =  JSON.stringify(content)
    var a = document.createElement("a");
    var file = new Blob([content], {type: contentType});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
}


async function listen(){
  if (specGenerator.isListening()) {
    specGenerator.stopListening();
    document.getElementById('listen').textContent = 'Listen for "OK ATLAS"';
    return;
  }
  document.getElementById('listen').textContent = 'Stop';
  specGenerator.listen(async(vals) => {
    tf.engine().startScope();
    const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
    let probs;
    if (TIMESTEPS != 0) {
      const split = tf.split(input, TIMESTEPS, 1);
      const stack = tf.stack(split, 1);
      probs = model.predict(stack);
    }
    else {
      probs = model.predict(input);
    }
    const confidence = await probs.data();
    console.log("confidence:", confidence[0].toFixed(4));
    await highlight(confidence[0]);

  // save false positives
   if (confidence[0] >= threshold && fpMode){
    let timestamp = getTime();
    console.log("False_positive", confidence[0]);
    download(vals, "falsePositive_"+timestamp+".json", 'text/plain');
  }
    tf.dispose([input, probs]);

   tf.engine().endScope();
  }, {
    fftSize : fftSizeValue,
    sampleRateHz : sampleRateHzValue,
    overlapFactor: overlapFactorThreshold,
    numFramesPerSpectrogram: numFramesPerSpectrogramValue, 
    columnTruncateLength:columnTruncateLengthValue,
    frameInterval:frameIntervalValue
  });
} 

async function toggleFPMode(){
  fpMode = !fpMode;
  let element = document.getElementById('fp_mode');
  element.innerHTML = "False Positive mode:<br/>" + fpMode;
  element.classList.toggle('button_critical_active');
}

function flatten(tensors) {
 const size = tensors[0].length;
 const result = new Float32Array(tensors.length * size);
 tensors.forEach((arr, i) => result.set(arr, i * size));
 return result;
}

function normalize(x) {
 const mean = -100;
 const std = 10;
 return x.map(x => (x - mean) / std);
}

var updateChart = function(value) {
  chartData.push({
    x: xVal,
    y: value
  })
  xVal ++;

  if (chartData.length > 20) {
    chartData.shift();
  }

  chart.render();
}
/* When the user clicks on the button,
toggle between hiding and showing the dropdown content */
function showDropdown() {
  document.getElementById("myDropdown").classList.toggle("show");
}

// Close the dropdown menu if the user clicks outside of it
window.onclick = function(event) {
  if (!event.target.matches('.dropbtn')) {
    var dropdowns = document.getElementsByClassName("dropdown-content");
    var i;
    for (i = 0; i < dropdowns.length; i++) {
      var openDropdown = dropdowns[i];
      if (openDropdown.classList.contains('show')) {
        openDropdown.classList.remove('show');
      }
    }
  }
} 

function addZero(x, n) {
  while (x.toString().length < n) {
      x = "0" + x;
  }
  return x;
}

function getTime() {
  let h = addZero(date.getHours(), 2);
  let m = addZero(date.getMinutes(), 2);
  let s = addZero(date.getSeconds(), 2);
  return h + ":" + m + ":" + s;
}