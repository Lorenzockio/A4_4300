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

  return vec4f(floor(A*255.),0.,floor(B*255.), 1.);
}
