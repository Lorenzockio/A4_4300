import { default as seagulls } from './gulls.js'

const frag = `
@group(0) @binding(0) var<uniform> res:   vec2f;
@group(0) @binding(1) var<storage> state: array<f32>;

fn index( x:i32, y:i32 ) -> u32 {
  let _res = vec2i(res);
  return u32( ((y % _res.y) * _res.x + ( x % _res.x )) * 2 );
}

@fragment 
fn fs( @builtin(position) pos : vec4f ) -> @location(0) vec4f {
  let x = i32(pos.x);
  let y = i32(pos.y);
  let i = index(x, y);

  let A = state[i];
  let B = state[i + 1]; 

  return vec4f(B+pos.x/res.x,A,A, 1.);
}
`

const compute = `
@group(0) @binding(0) var<uniform> res: vec2f;
@group(0) @binding(1) var<storage> statein: array<f32>;
@group(0) @binding(2) var<storage, read_write> stateout: array<f32>;
@group(0) @binding(3) var<uniform> mouseX: i32;
@group(0) @binding(4) var<uniform> mouseY: i32;
@group(0) @binding(5) var<uniform> feed: f32;
@group(0) @binding(6) var<uniform> kill: f32;
@group(0) @binding(7) var<uniform> dA: f32;
@group(0) @binding(8) var<uniform> dB: f32;
@group(0) @binding(9) var<uniform> LRA: f32;
@group(0) @binding(10) var<uniform> UDA: f32;
@group(0) @binding(11) var<uniform> LRB: f32;
@group(0) @binding(12) var<uniform> UDB: f32;



fn index( x:i32, y:i32 ) -> u32 {
  let _res = vec2i(res);
  return u32( ((y % _res.y) * _res.x + ( x % _res.x )) * 2 );
}

fn laplaceA( x:i32, y:i32) -> f32 {
  let lap=
    statein[ index(x - 1, y) ] * (0.2 - LRA) + 
    statein[ index(x + 1, y) ] * (0.2 + LRA)+ 
    statein[ index(x, y - 1) ] * (0.2 - UDA) + 
    statein[ index(x, y + 1) ] * (0.2 + UDA) +
    statein[ index(x - 1, y - 1) ] * 0.05 + 
    statein[ index(x + 1, y - 1) ] * 0.05 +
    statein[ index(x + 1, y + 1) ] * 0.05 + 
    statein[ index(x - 1, y + 1) ] * 0.05 -
    statein[ index(x, y) ];
  return lap;
}

fn laplaceB( x:i32, y:i32) -> f32 {
  let lap=
    statein[ index(x - 1, y) + 1] * (0.2 + LRB) + 
    statein[ index(x + 1, y) + 1] * (0.2 - LRB) + 
    statein[ index(x, y - 1) + 1] * (0.2 + UDB) + 
    statein[ index(x, y + 1) + 1] * (0.2 - UDB) +
    statein[ index(x - 1, y - 1) + 1] * 0.05 + 
    statein[ index(x + 1, y - 1) + 1] * 0.05 +
    statein[ index(x + 1, y + 1) + 1] * 0.05 + 
    statein[ index(x - 1, y + 1) + 1] * 0.05 -
    statein[ index(x, y) + 1];
  return lap;
}
@compute
@workgroup_size(8,8)
fn cs( @builtin(global_invocation_id) _cell:vec3u ) {
  let cell = vec3i(_cell);

  let i = index(cell.x, cell.y);
  let A = statein[i];
  let B = statein[i + 1];

  var outA = A + (dA * laplaceA(cell.x, cell.y)) - (A * B * B) + (feed * (1.-A));

  var outB = B + (dB * laplaceB(cell.x, cell.y)) + (A * B * B) - (kill + feed) * B;

  stateout[i] = outA;
  stateout[i + 1] = outB;

  var mdex = index( mouseX, mouseY );
  if (i == mdex || i == mdex - 2 || i == mdex + 2) {
    stateout[i] = 0.0;
    stateout[i + 1] = 1.0;
  }
}
`

const sg      = await seagulls.init(),
      render  = seagulls.constants.vertex + frag,
      size    = (window.innerWidth * window.innerHeight),
      state   = new Float32Array( size * 2)

for( let i = 0; i < size; i++ ) {
  if (state[ i * 2+ 1] === 0.0) {
    state[ i * 2] = 1.0;
  } else {
    state[ i * 2] = 0.0;
  }
}

const statebuffer1 = sg.buffer( state )
const statebuffer2 = sg.buffer( state )
const res = sg.uniform([ window.innerWidth, window.innerHeight ])
let mouseX = sg.uniform(Math.random() * window.innerWidth)
let mouseY = sg.uniform(Math.random() * window.innerHeight)

const feed = document.querySelector('#feed')
let feed_u = sg.uniform( feed.value )
const kill = document.querySelector('#kill')
let kill_u = sg.uniform( kill.value )
const dA = document.querySelector('#dA')
let dA_u = sg.uniform( dA.value )
const dB = document.querySelector('#dB')
let dB_u = sg.uniform( dB.value )
const LRA = document.querySelector('#LR')
let LRA_u = sg.uniform( LRA.value )
const UDA = document.querySelector('#UD')
let UDA_u = sg.uniform( UDA.value )
const LRB = document.querySelector('#LRB')
let LRB_u = sg.uniform( LRB.value )
const UDB = document.querySelector('#UDB')
let UDB_u = sg.uniform( UDB.value )
const resetButton = document.querySelector('#reset')


const renderPass = await sg.render({
  shader: render,
  data: [
    res,
    sg.pingpong( statebuffer1, statebuffer2 )
  ]
})

const computePass = sg.compute({
  shader: compute,
  data: [ 
    res, 
    sg.pingpong( statebuffer1, statebuffer2 ), 
    mouseX, 
    mouseY, 
    feed_u, 
    kill_u, 
    dA_u, 
    dB_u, 
    LRA_u, 
    UDA_u, 
    LRB_u, 
    UDB_u 
  ],
  dispatchCount:  [Math.round(seagulls.width / 8), Math.round(seagulls.height/8), 1],
  times: 10
})

feed.oninput = ()=> feed_u.value = parseFloat( feed.value )
kill.oninput = ()=> kill_u.value = parseFloat( kill.value )
dA.oninput = ()=> dA_u.value = parseFloat( dA.value )
dB.oninput = ()=> dB_u.value = parseFloat( dB.value )
LRA.oninput = ()=> LRA_u.value = parseFloat( LRA.value )
UDA.oninput = ()=> UDA_u.value = parseFloat( UDA.value )
LRB.oninput = ()=> LRB_u.value = parseFloat( LRB.value )
UDB.oninput = ()=> UDB_u.value = parseFloat( UDB.value )
resetButton.onclick = () => {
  location.reload()
};

const canvas = document.getElementById("canvas");
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

canvas.addEventListener("click", (event) => {
    const x = event.offsetX;
    const y = event.offsetY;
    //This does not set correct position. I gave up
    mouseX.value = x;
    mouseY.value = y;
});

sg.run( computePass, renderPass )
