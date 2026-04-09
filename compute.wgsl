@group(0) @binding(0) var<uniform> res: vec2f;
@group(0) @binding(1) var<storage> statein: array<f32>;
@group(0) @binding(2) var<storage, read_write> stateout: array<f32>;

fn index( x:i32, y:i32 ) -> u32 {
  let _res = vec2i(res);
  return u32( ((y % _res.y) * _res.x + ( x % _res.x )) * 2 );
}

@compute
@workgroup_size(8,8)
fn cs( @builtin(global_invocation_id) _cell:vec3u ) {
  let cell = vec3i(_cell);

  let i = index(cell.x, cell.y);
  let A = statein[i];
  let B = statein[i + 1];

  outA = A + 
    (dA * laplaceA() * A) - 
    (A * B * B) +
    (feed * (1-A));

  outA = B +
    (dB * laplaceB() * B) + 
    (A * B * B) -
    (k + feed) * B;

  stateout[i] = statein[i];
}
