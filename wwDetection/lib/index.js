let recognizer,baseRecognizer;
let model;
let threshold = 0.99999;

async function app() {
 baseRecognizer = speechCommands.create('BROWSER_FFT');
 await baseRecognizer.ensureModelLoaded();
 let modelName = 'atlas_model_v4';
 model = await tf.loadLayersModel('https://awahlang.github.io/wwDetection/models/'+modelName+'/model.json');
 console.log("model used:", modelName);
}

app().then((result)=>{
  document.getElementById('listen').disabled = false;
  document.getElementById('predictions').style.display = 'flex';
});

async function highlight(confidence) {
  pred_divs = document.getElementsByClassName('prediction');
  for(let i=0;i<pred_divs.length;i++){
    pred_divs[i].classList.remove('green_background');
  }
  if((confidence >= threshold)){
    document.getElementById('ok_atlas').innerHTML = 'ok_Atlas<br>' + confidence.toFixed(5);
    document.getElementById('ok_atlas').classList.add('green_background');
   // document.getElementById('yes').play();
  }
  else {
    let val = 1.000 - confidence.toFixed(5);
    document.getElementById('other').innerHTML = 'other<br>' + val;
    document.getElementById('other').classList.add('green_background');
  }
}

function setThreshold(){
  let value = document.getElementById("threshold").value;
  console.log("update threshold: ",value);
  threshold = value;
}

const NUM_FRAMES = 43;
const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
const words = {
  0:'_background_noise',
  1:'negative',
  2:'ok_atlas',
};

function download(content, fileName, contentType) {
    content =  JSON.stringify(content)
    var a = document.createElement("a");
    var file = new Blob([content], {type: contentType});
    a.href = URL.createObjectURL(file);
    a.download = fileName;
    a.click();
}

// const words = ['hey_atlas','negative_word','noise'];
let prob;
async function listen(){
  if (baseRecognizer.isListening()) {
   baseRecognizer.stopListening();
   document.getElementById('listen').textContent = 'Listen for "OK ATLAS"';
   return;
  }
  document.getElementById('listen').textContent = 'Stop';
  baseRecognizer.listen(async ({spectrogram: {frameSize, data}}) => {
  const vals = normalize(data.subarray(-frameSize * NUM_FRAMES)); // (232 * 43)
   
   const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
   const probs = model.predict(input);
   // const predLabel = probs.argMax(1);
   // const label = (await predLabel.data())[0];
   // const confidence = probs.max(1);
   // const conf = (await confidence.data())[0];
   // console.log(label+","+words[label]+","+conf);
   const confidence = await probs.data()
   console.log("confidence:", confidence[0].toFixed(4))
   await highlight(confidence[0]);
   tf.dispose([input, probs]);
  }, {
   overlapFactor: 0.50,
   includeSpectrogram: true,
   invokeCallbackOnNoiseAndUnknown: false
  });
}

app();

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
