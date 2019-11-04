
let data;
let model;
let xs, ys;
let chosenColor = [255, 255, 255];
let labelList = [
    'red-ish',
    'green-ish',
    'blue-ish',
    'orange-ish',
    'yellow-ish',
    'pink-ish',
    'purple-ish',
    'brown-ish',
    'grey-ish'
]
let pIndex = 0;
let trained = false;
let lastLoss;
let fitCallbacks;
function preload() {
    data = loadJSON('colorData.json')
}
function setup() {
    createCanvas(window.innerWidth, window.innerHeight)
    let colors = [];
    let labels = [];
    for (let record of data.entries) {
        let col = [record.r / 255, record.g / 255, record.b / 255];
        colors.push(col);
        labels.push(labelList.indexOf(record.label));
    }
    // loadCharts() // -- Charts preset. HEAVY AH!!!
    xs = tf.tensor2d(colors);
    let labelsTensor = tf.tensor1d(labels, 'int32');
    ys = tf.oneHot(labelsTensor, 9);
    labelsTensor.dispose();
    model = tf.sequential();
    let hidden = tf.layers.dense({
        units: 16,
        activation: 'sigmoid',
        inputDim: 3
    });
    let output = tf.layers.dense({
        units: 9,
        activation: 'softmax'
    });
    model.add(hidden);
    model.add(output);
    const lr = 0.2
    const optimizer = tf.train.sgd(lr);

    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy'
    });
}
function predict(color1, color2, color3) {
    if (!trained) {
        return console.warn('Model is not trained yet! Use train(epochs); to train the model.');
    }
    chosenColor = [color1, color2, color3];
    tf.tidy(() => {
    pIndex = tf.argMax(model.predict(tf.tensor2d([chosenColor])), axis = 1).dataSync()[0];
    });
    console.log('Uhm.. That\'s kinda ' + labelList[pIndex]);
}
function train(epochs, isSafe) {
    let fepochs = 1;
    if (!epochs) {
        console.warn('Got no epochs! Defaults to 1');
        epochs = 1;
    }
    console.log('Training started! It\'s gonna take a while, so please wait..')
    if (isSafe) { // isSafe is experimental. It slows your training but finds a good epoch rate;
        fepochs = epochs;
        epochs = 1
    }
    const options = {
        epochs: epochs,
        validationSplit: 0.2,
        shuffle: true,
        callbacks: fitCallbacks
    }
    train_start(fepochs, options, isSafe);
}
async function train_start(fepochs, options, isSafe) {
    for (let i = 0; i < fepochs; i++) {
        await model.fit(xs, ys, options).then((results) => {
            trained = true;
            if (isSafe) {
                if (results.history.loss[0] > lastLoss) {
                    lastLoss = 100;
                    console.log('Training forcibly stopped. Current safe-loss is ' + results.history.loss[0] + '. You should set your epoch amount to ' + (i - 1));
                    i = fepochs - 1;
                } else {
                    lastLoss = results.history.loss[0];
                }
            } else {
                console.log('Training finished! Current loss is ' + results.history.loss[results.history.loss.length - 1] + '. Now you can use predict([r,g,b]);');
            }
        });
    }
    if (isSafe) {
        console.log('Training finished! Now you can use predict([r,g,b]);');
    }
}

function draw() {
    background(chosenColor);
    textSize(16 * ((width / 400)));
    textAlign(CENTER, BASELINE);
    if (pIndex == 0) {
        text('NN Color Predictor with Tensorflow.js', width / 2, height / 2);
    } else {
        text(labelList[pIndex], width / 2, height / 2);
    }
    textSize(8 * ((width / 400)));
    textAlign(CENTER, BOTTOM);
    text('Use train(epochs, isSafe); and predict(r, g, b); in console.\n- isSafe calculates a good epoch rate to use but makes everything work much slower!\n\nCrowdsourced by The Coding Train Community', width / 2, height / 1.25);
}

function loadCharts() {
    const visor = tfvis.visor()
    const surface = {
        "TrainingData": visor.surface({ name: 'Training Data', tab: 'Data' }),
        "RGBDifference": visor.surface({ name: 'RGB Difference', tab: 'Data' })
    };
    let cr = 0;
    let cg = 0;
    let cb = 0;
    for (i = 1; i < data.entries.length; i++) {
        cr += data.entries[i].r;
        cg += data.entries[i].g;
        cb += data.entries[i].b;
    }
    cr = cr / data.entries.length;
    cg = cg / data.entries.length;
    cb = cb / data.entries.length;
    const headers = [
        'Col 1',
        'Col 2',
        'Col 3',
    ];

    tfvis.render.barchart(surface["RGBDifference"],
        [ // min/max*100
            { index: 'Red', value: cr / (cg - cb) },
            { index: 'Green', value: cg / (cr - cb) },
            { index: 'Blue', value: cb / (cg - cr) }
        ],
        {
            xLabel: 'Color',
            yLabel: 'Average'
        }
    );
    fitCallbacks = tfvis.show.fitCallbacks({
        name: 'Statistics',
        tab: 'Training'
    },
        [
            'loss',
            'val_loss',
            'acc',
            'val_acc'
        ]);
}