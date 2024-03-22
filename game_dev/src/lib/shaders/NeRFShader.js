export let NeRFShader = {
    gbuffer_vert: `

    out vec2 vUv;
    out vec3 vPosition;
    out vec3 rayDirection;


    void main() {
        vUv = uv;
        vPosition = position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
        rayDirection = (modelMatrix * vec4( position, 1.0 )).rgb - cameraPosition;
    }
    `,
    gbuffer_vert_wind: `

    out vec2 vUv;
    out vec3 vPosition;
    out vec3 rayDirection;

    uniform float time;

    void main() {
        vUv = uv;
        vec3 m_position = vec3(position.x + 0.05 * sin(time) * exp(position.y-1.0) * exp(position.y-1.0), position.y, position.z + 0.05 * sin(time) * exp(position.y-1.0) * exp(position.y-1.0));
        vPosition = m_position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4( m_position, 1.0 );
        rayDirection = (modelMatrix * vec4( m_position, 1.0 )).rgb - cameraPosition;
    }
    `,
    vertexShaderDepth: `
    out vec3 vPosition;
    out vec3 rayDirection;

    uniform float time;

    void main() {
        vec3 m_position = vec3(position.x + 0.05 * sin(time) * exp(position.y-1.0) * exp(position.y-1.0), position.y, position.z + 0.05 * sin(time) * exp(position.y-1.0) * exp(position.y-1.0));
        vPosition = m_position;
        gl_Position = projectionMatrix * modelViewMatrix * vec4( m_position, 1.0 );
    }
    `,
    fragmentShaderDepth: `
    
    #include <common>
    #include <packing>
    #include <uv_pars_fragment>
    #include <map_pars_fragment>
    #include <alphamap_pars_fragment>
    #include <logdepthbuf_pars_fragment>
    #include <clipping_planes_pars_fragment>

    void main() {

        #include <clipping_planes_fragment>

        vec4 diffuseColor = vec4( 1.0 );

        #include <map_fragment>
        #include <alphamap_fragment>
        #include <alphatest_fragment>

        #include <logdepthbuf_fragment>

        gl_FragColor = packDepthToRGBA( gl_FragCoord.z );
    }
    `,
    gbuffer_frag: `
    precision mediump float;

    in vec2 vUv;
    in vec3 vPosition;
    in vec3 rayDirection;

    out vec4 pc_FragColor;

    uniform mediump sampler2D tDiffuse0;
    uniform mediump sampler2D tDiffuse1;

    uniform mediump sampler2D weightsZero;
    uniform mediump sampler2D weightsOne;
    uniform mediump sampler2D weightsTwo;

    mediump vec3 evaluateNetwork( mediump vec4 f0, mediump vec4 f1, mediump vec4 viewdir) {
        mediump float intermediate_one[NUM_CHANNELS_ONE] = float[](
            BIAS_LIST_ZERO
        );
        for (int j = 0; j < NUM_CHANNELS_ZERO; ++j) {
            mediump float input_value = 0.0;
            if (j < 4) {
            input_value =
                (j == 0) ? f0.r : (
                (j == 1) ? f0.g : (
                (j == 2) ? f0.b : f0.a));
            } else if (j < 8) {
            input_value =
                (j == 4) ? f1.r : (
                (j == 5) ? f1.g : (
                (j == 6) ? f1.b : f1.a));
            } else {
            input_value =
                (j == 8) ? viewdir.r : (
                (j == 9) ? -viewdir.b : viewdir.g); //switch y-z axes
            }
            for (int i = 0; i < NUM_CHANNELS_ONE; ++i) {
            intermediate_one[i] += input_value *
                texelFetch(weightsZero, ivec2(j, i), 0).x;
            }
        }
        mediump float intermediate_two[NUM_CHANNELS_TWO] = float[](
            BIAS_LIST_ONE
        );
        for (int j = 0; j < NUM_CHANNELS_ONE; ++j) {
            if (intermediate_one[j] <= 0.0) {
                continue;
            }
            for (int i = 0; i < NUM_CHANNELS_TWO; ++i) {
                intermediate_two[i] += intermediate_one[j] *
                    texelFetch(weightsOne, ivec2(j, i), 0).x;
            }
        }
        mediump float result[NUM_CHANNELS_THREE] = float[](
            BIAS_LIST_TWO
        );
        for (int j = 0; j < NUM_CHANNELS_TWO; ++j) {
            if (intermediate_two[j] <= 0.0) {
                continue;
            }
            for (int i = 0; i < NUM_CHANNELS_THREE; ++i) {
                result[i] += intermediate_two[j] *
                    texelFetch(weightsTwo, ivec2(j, i), 0).x;
            }
        }
        for (int i = 0; i < NUM_CHANNELS_THREE; ++i) {
            result[i] = 1.0 / (1.0 + exp(-result[i]));
        }
        return vec3(result[0]*viewdir.a+(1.0-viewdir.a),
                    result[1]*viewdir.a+(1.0-viewdir.a),
                    result[2]*viewdir.a+(1.0-viewdir.a));
    }

    void main() {
        // write color to G-Buffer
        vec4 gColor1 = texture( tDiffuse0, vUv );
        if (gColor1.r == 0.0) discard;
        vec4 gColor0 = vec4( normalize(rayDirection), 1.0 );
        vec4 gColor2 = texture( tDiffuse1, vUv );
        if(gColor0.a < 0.6) discard;
        //pc_FragColor.rgb  = diffuse1.rgb;
        pc_FragColor.rgb = evaluateNetwork(gColor1,gColor2,gColor0);
        pc_FragColor.a = 1.0;
    }
    `,
    sgVertexShaderSource: `
    attribute vec3 _byte_normal;
    
    attribute vec3 _sg_mean_0;
    attribute vec3 _sg_mean_1;
    attribute vec3 _sg_mean_2;

    attribute float _sg_scale_0;
    attribute float _sg_scale_1;
    attribute float _sg_scale_2;

    attribute vec3 _sg_color_0;
    attribute vec3 _sg_color_1;
    attribute vec3 _sg_color_2;

    varying vec3 vDiffuse;
    varying vec3 vPosition;

    varying vec3 vSgMean0;
    varying vec3 vSgMean1;
    varying vec3 vSgMean2;

    varying float vSgScale0;
    varying float vSgScale1;
    varying float vSgScale2;

    varying vec3 vSgColor0;
    varying vec3 vSgColor1;
    varying vec3 vSgColor2;

    uniform mat3 worldspace_R_opengl;
    
    void main() {
        vec3 positionWorld = position;
        vPosition = position;
        vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(positionWorld, 1.0);
        gl_Position = positionClip;
        positionClip /= positionClip.w;

        vDiffuse = color.rgb;

        vSgMean0 = _sg_mean_0 * (2.0 / 255.0) - 1.0;
        vSgMean1 = _sg_mean_1 * (2.0 / 255.0) - 1.0;
        vSgMean2 = _sg_mean_2 * (2.0 / 255.0) - 1.0;

        vSgScale0 = 5.0 * _sg_scale_0 / 255.0;
        vSgScale1 = 5.0 * _sg_scale_1 / 255.0;
        vSgScale2 = 5.0 * _sg_scale_2 / 255.0;

        vSgColor0 = _sg_color_0 / 255.0;
        vSgColor1 = _sg_color_1 / 255.0;
        vSgColor2 = _sg_color_2 / 255.0;
    }
    `,
    sgFragmentShaderSource: `

    varying vec3 vDiffuse;
    varying vec3 vPosition;

    varying vec3 vSgMean0;
    varying vec3 vSgMean1;
    varying vec3 vSgMean2;

    varying float vSgScale0;
    varying float vSgScale1;
    varying float vSgScale2;

    varying vec3 vSgColor0;
    varying vec3 vSgColor1;
    varying vec3 vSgColor2;

    uniform mat4 gmatrix_inv;
    uniform mat3 intrinsic;
    uniform mat3 c2w_T;

    uniform float fx;
    uniform float fy;
    uniform float cx;
    uniform float cy;
    uniform mat3 worldspace_R_opengl;


    vec3 evalSphericalGaussian(vec3 direction, vec3 mean, float scale, vec3 color) {
        return color * exp(scale * (dot(direction, mean) - 1.0));
    }

    void main() {
        vec3 diffuse = vDiffuse;
        
        vec4 positionCamWorld = gmatrix_inv * vec4(vPosition, 1.0);
        positionCamWorld /= positionCamWorld.w;
        vec3 positionScrWorld = intrinsic * vec3(positionCamWorld.xyz);
        positionScrWorld /= positionScrWorld.z;
        
        float px = 2.0 * cx - positionScrWorld.x;
        float py = positionScrWorld.y;
        float _px = (px - cx) / fx;
        float _py = (py - cy) / fy;
        vec3 directionCamWorld = vec3(_px, _py, 1);

        vec3 directionWorld = directionCamWorld * c2w_T;

        vec3 viewDependence = evalSphericalGaussian(
            directionWorld, normalize(vSgMean0), vSgScale0, vSgColor0);
        viewDependence += evalSphericalGaussian(
            directionWorld, normalize(vSgMean1), vSgScale1, vSgColor1);
        viewDependence += evalSphericalGaussian(
            directionWorld, normalize(vSgMean2), vSgScale2, vSgColor2);

        vec3 color;
        color = diffuse + viewDependence;
        // color = vec3(1.0, 0.0, 0.0);
        gl_FragColor = vec4(color, 1.0);
    }
    `,
    sgVertexShaderSource_diffuse: `
    attribute vec3 _byte_normal;

    varying vec3 vDiffuse;

    
    void main() {
        vec3 positionWorld = position;
        vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(positionWorld, 1.0);
        gl_Position = positionClip;
        positionClip /= positionClip.w;

        vDiffuse = color.rgb;

    }
    `,
    sgFragmentShaderSource_diffuse: `

    varying vec3 vDiffuse;


    void main() {

        vec3 diffuse = vDiffuse;
    
        vec3 color;
        color = diffuse;
        
        gl_FragColor = vec4(color, 1.0);
    }
    `,
    sgVertexShaderSource_1sp: `
    attribute vec3 _byte_normal;
    
    attribute vec3 _sg_mean_0;

    attribute float _sg_scale_0;

    attribute vec3 _sg_color_0;

    varying vec3 vDiffuse;
    varying vec3 vPosition;

    varying vec3 vSgMean0;

    varying float vSgScale0;

    varying vec3 vSgColor0;

    uniform mat3 worldspace_R_opengl;

        
    void main() {
        vec3 positionWorld = position;
        vPosition = position;
        vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(positionWorld, 1.0);
        gl_Position = positionClip;
        positionClip /= positionClip.w;
        
        vDiffuse = color.rgb;

        vSgMean0 = _sg_mean_0 * (2.0 / 255.0) - 1.0;

        vSgScale0 = 5.0 * _sg_scale_0 / 255.0;

        vSgColor0 = _sg_color_0 / 255.0;
    }
    `,
    sgFragmentShaderSource_1sp: `

    varying vec3 vDiffuse;
    varying vec3 vPosition;

    varying vec3 vSgMean0;

    varying float vSgScale0;

    varying vec3 vSgColor0;

    
    uniform mat4 gmatrix_inv;
    uniform mat3 intrinsic;
    uniform mat3 c2w_T;

    uniform float fx;
    uniform float fy;
    uniform float cx;
    uniform float cy;
    uniform mat3 worldspace_R_opengl;


    vec3 evalSphericalGaussian(vec3 direction, vec3 mean, float scale, vec3 color) {
        return color * exp(scale * (dot(direction, mean) - 1.0));
    }

    void main() {


        vec4 positionCamWorld = gmatrix_inv * vec4(vPosition, 1.0);
        positionCamWorld /= positionCamWorld.w;
        vec3 positionScrWorld = intrinsic * vec3(positionCamWorld.xyz);
        positionScrWorld /= positionScrWorld.z;
        
        float px = 2.0 * cx - positionScrWorld.x;
        float py = positionScrWorld.y;
        float _px = (px - cx) / fx;
        float _py = (py - cy) / fy;
        vec3 directionCamWorld = vec3(_px, _py, 1);

        vec3 directionWorld = directionCamWorld * c2w_T;
        
        
        vec3 diffuse = vDiffuse;

        vec3 viewDependence = evalSphericalGaussian(
            directionWorld, (vSgMean0), vSgScale0, vSgColor0);
        
    
        // vec3 color = vec3(positionCamWorld.x, positionCamWorld.y, positionCamWorld.z);
        // vec3 color = vec3(0, px / 1408.0, py / 376.0);
        // vec3 color = (directionWorld + 1.0) / 2.0;
        vec3 color = diffuse + viewDependence;
        
        gl_FragColor = vec4(color, 1.0);
    }
    `,
    RenderVertShader: `

    varying vec2 vUv;
    varying vec3 vPosition;


    uniform mat4 gmatrix_inv;
    uniform mat3 intrinsic;
    uniform mat3 c2w_T;

    uniform float fx;
    uniform float fy;
    uniform float cx;
    uniform float cy;

    void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
        vPosition = position;
    }
    `,
    RenderFragShader_template: `
    precision highp float;

    varying vec2 vUv;
    varying vec3 vPosition;

    uniform int mode;

    uniform highp sampler2D tDiffuse;
    uniform highp sampler2D tSpecular;

    uniform highp sampler2D weightsZero;
    uniform highp sampler2D weightsOne;


    uniform mat4 gmatrix_inv;
    uniform mat3 intrinsic;
    uniform mat3 c2w_T;

    uniform float fx;
    uniform float fy;
    uniform float cx;
    uniform float cy;

    float inputFetch(vec4 f0, vec3 viewdir, int j) {
        float input_value = 0.0;
        if (j < 4) {
            input_value = (j == 0) ? viewdir.r : ((j == 1) ? viewdir.g : ((j == 2) ? viewdir.b : f0.r));
        } else {
            input_value = (j == 4) ? f0.g : ((j == 5) ? f0.b : f0.a);
        }
        // if (abs(input_value) < 0.1 / 255.0) {
        //     input_value = 0.0;
        // }
        return input_value;
    }

    vec3 evaluateNetwork(vec4 f0, vec3 viewdir) {

        // NUM_CHANNELS_ZERO (input_dim) is hard-coded as 6
        // NUM_CHANNELS_ONE (hidden_dim) can vary, but should be divisible by 4
        // NUM_CHANNELS_TWO (output_dim) is hard-coded as 3
        
        vec4 v;
        mat4 w;

        // first layer: 6 --> NUM_CHANNELS_ONE

        vec4 result_one[NUM_CHANNELS_ONE / 4];

        v = vec4(
            inputFetch(f0, viewdir, 0),
            inputFetch(f0, viewdir, 1),
            inputFetch(f0, viewdir, 2),
            inputFetch(f0, viewdir, 3)
        );

        for (int i = 0; i < NUM_CHANNELS_ONE; i += 4) {
            w = mat4(
                texelFetch(weightsZero, ivec2(0, i), 0),
                texelFetch(weightsZero, ivec2(0, i + 1), 0),
                texelFetch(weightsZero, ivec2(0, i + 2), 0),
                texelFetch(weightsZero, ivec2(0, i + 3), 0)
            );
            result_one[i / 4] += v * w;
        }

        v = vec4(
            inputFetch(f0, viewdir, 4),
            inputFetch(f0, viewdir, 5),
            0.0,
            0.0
        );

        for (int i = 0; i < NUM_CHANNELS_ONE; i += 4) {
            w = mat4(
                texelFetch(weightsZero, ivec2(0, NUM_CHANNELS_ONE + i), 0),
                texelFetch(weightsZero, ivec2(0, NUM_CHANNELS_ONE + i + 1), 0),
                texelFetch(weightsZero, ivec2(0, NUM_CHANNELS_ONE + i + 2), 0),
                texelFetch(weightsZero, ivec2(0, NUM_CHANNELS_ONE + i + 3), 0)
            );
            result_one[i / 4] += v * w;
        }

        // second layer: NUM_CHANNELS_ONE --> 3

        vec3 result;

        for (int i = 0; i < NUM_CHANNELS_ONE / 4; i++) {
            v = max(result_one[i], 0.0); // relu
            w = mat4(
                texelFetch(weightsOne, ivec2(0, i * 3), 0),
                texelFetch(weightsOne, ivec2(0, i * 3 + 1), 0),
                texelFetch(weightsOne, ivec2(0, i * 3 + 2), 0),
                vec4(0.0) // padding
            );
            result += (v * w).xyz;
        }

        // sigmoid
        return 1.0 / (1.0 + exp(-result)); 
    }

    void main() {    
        vec4 diffuse = texture( tDiffuse, vUv );

        vec4 positionCamWorld = gmatrix_inv * vec4(vPosition, 1.0);
        positionCamWorld /= positionCamWorld.w;
        vec3 positionScrWorld = intrinsic * vec3(positionCamWorld.xyz);
        positionScrWorld /= positionScrWorld.z;
        
        float px = 2.0 * cx - positionScrWorld.x;
        float py = positionScrWorld.y;
        float _px = (px - cx) / fx;
        float _py = (py - cy) / fy;
        vec3 directionCamWorld = vec3(_px, _py, 1);

        vec3 directionWorld = directionCamWorld * c2w_T;

        vec3 rayDirection = directionWorld;

        if (mode == 1) { // diffuse
            gl_FragColor.rgb = diffuse.rgb;
        } else {
            vec4 specular = texture( tSpecular, vUv );
            if (mode == 2) { // specular
                gl_FragColor.rgb = evaluateNetwork(specular, normalize(rayDirection));
            } else { // full
                // gl_FragColor.rgb = clamp(normalize(rayDirection), 0.0f, 1.0f);
                // gl_FragColor.rgb = clamp((normalize(rayDirection) + 1.0f) / 2.0f, 0.0f, 1.0f);
                gl_FragColor.rgb = clamp(diffuse.rgb + evaluateNetwork(specular, normalize(rayDirection)), 0.0f, 1.0f);
                // gl_FragColor.rgb = clamp(evaluateNetwork(specular, normalize(rayDirection)), 0.0f, 1.0f);
            }
        }
        gl_FragColor.a = 1.0;
    }
    `,
    RenderFragShader_5sp_template: `
    precision highp float;

    varying vec2 vUv;
    varying vec3 vPosition;

    uniform int mode;

    uniform highp sampler2D tDiffuse;
    uniform highp sampler2D tSpecular;

    uniform highp sampler2D weightsZero;
    uniform highp sampler2D weightsOne;


    uniform mat4 gmatrix_inv;
    uniform mat3 intrinsic;
    uniform mat3 c2w_T;

    uniform float fx;
    uniform float fy;
    uniform float cx;
    uniform float cy;

    float inputFetch(vec4 af0, vec4 f0, vec3 viewdir, int j) {
        float input_value = 0.0;
        if (j < 4) {
            input_value = (j == 0) ? viewdir.r : ((j == 1) ? viewdir.g : ((j == 2) ? viewdir.b : af0.a));
        } else {
            input_value = (j == 4) ? f0.r : ((j == 5) ? f0.g : ((j == 6) ? f0.b : f0.a) );
        }
        // if (abs(input_value) < 0.1 / 255.0) {
        //     input_value = 0.0;
        // }
        return input_value;
    }

    vec3 evaluateNetwork(vec4 af0, vec4 f0, vec3 viewdir) {

        // NUM_CHANNELS_ZERO (input_dim) is hard-coded as 6
        // NUM_CHANNELS_ONE (hidden_dim) can vary, but should be divisible by 4
        // NUM_CHANNELS_TWO (output_dim) is hard-coded as 3
        
        vec4 v;
        mat4 w;

        // first layer: 5+3 --> NUM_CHANNELS_ONE

        vec4 result_one[NUM_CHANNELS_ONE / 4];

        v = vec4(
            inputFetch(af0, f0, viewdir, 0),
            inputFetch(af0, f0, viewdir, 1),
            inputFetch(af0, f0, viewdir, 2),
            inputFetch(af0, f0, viewdir, 3)
        );

        for (int i = 0; i < NUM_CHANNELS_ONE; i += 4) {
            w = mat4(
                texelFetch(weightsZero, ivec2(0, i), 0),
                texelFetch(weightsZero, ivec2(0, i + 1), 0),
                texelFetch(weightsZero, ivec2(0, i + 2), 0),
                texelFetch(weightsZero, ivec2(0, i + 3), 0)
            );
            result_one[i / 4] += v * w;
        }

        v = vec4(
            inputFetch(af0, f0, viewdir, 4),
            inputFetch(af0, f0, viewdir, 5),
            inputFetch(af0, f0, viewdir, 6),
            inputFetch(af0, f0, viewdir, 7)
        );

        for (int i = 0; i < NUM_CHANNELS_ONE; i += 4) {
            w = mat4(
                texelFetch(weightsZero, ivec2(0, NUM_CHANNELS_ONE + i), 0),
                texelFetch(weightsZero, ivec2(0, NUM_CHANNELS_ONE + i + 1), 0),
                texelFetch(weightsZero, ivec2(0, NUM_CHANNELS_ONE + i + 2), 0),
                texelFetch(weightsZero, ivec2(0, NUM_CHANNELS_ONE + i + 3), 0)
            );
            result_one[i / 4] += v * w;
        }

        // second layer: NUM_CHANNELS_ONE --> 3

        vec3 result;

        for (int i = 0; i < NUM_CHANNELS_ONE / 4; i++) {
            v = max(result_one[i], 0.0); // relu
            w = mat4(
                texelFetch(weightsOne, ivec2(0, i * 3), 0),
                texelFetch(weightsOne, ivec2(0, i * 3 + 1), 0),
                texelFetch(weightsOne, ivec2(0, i * 3 + 2), 0),
                vec4(0.0) // padding
            );
            result += (v * w).xyz;
        }

        // sigmoid
        return 1.0 / (1.0 + exp(-result)); 
    }

    void main() {    
        vec4 diffuse = texture( tDiffuse, vUv );

        vec4 positionCamWorld = gmatrix_inv * vec4(vPosition, 1.0);
        positionCamWorld /= positionCamWorld.w;
        vec3 positionScrWorld = intrinsic * vec3(positionCamWorld.xyz);
        positionScrWorld /= positionScrWorld.z;
        
        float px = 2.0 * cx - positionScrWorld.x;
        float py = positionScrWorld.y;
        float _px = (px - cx) / fx;
        float _py = (py - cy) / fy;
        vec3 directionCamWorld = vec3(_px, _py, 1);

        vec3 directionWorld = directionCamWorld * c2w_T;

        vec3 rayDirection = directionWorld;

        if (mode == 1) { // diffuse
            gl_FragColor.rgb = diffuse.rgb;
        } else {
            vec4 specular = texture( tSpecular, vUv );
            if (mode == 2) { // specular
                gl_FragColor.rgb = evaluateNetwork(diffuse, specular, normalize(rayDirection));
            } else { // full
                // gl_FragColor.rgb = rayDirection;
                // gl_FragColor.rgb = clamp((normalize(rayDirection) + 1.0f) / 2.0f, 0.0f, 1.0f);
                gl_FragColor.rgb = clamp(diffuse.rgb + evaluateNetwork(diffuse, specular, normalize(rayDirection)), 0.0f, 1.0f);
            }
        }
        gl_FragColor.a = 1.0;
    }
    `,
    sgVertexShaderSource_1sp_fullfloat: `
    attribute vec3 _byte_normal;
    
    attribute vec3 _sg_mean_0;

    attribute float _sg_scale_0;

    attribute vec3 _sg_color_0;

    varying vec3 vDiffuse;
    varying vec3 vPosition;

    varying vec3 vSgMean0;

    varying float vSgScale0;

    varying vec3 vSgColor0;

    uniform mat3 worldspace_R_opengl;

        
    void main() {
        vec3 positionWorld = position;
        vPosition = position;
        vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(positionWorld, 1.0);
        gl_Position = positionClip;
        positionClip /= positionClip.w;
        
        vDiffuse = color.rgb;

        vSgMean0 = _sg_mean_0;

        vSgScale0 = _sg_scale_0;

        vSgColor0 = _sg_color_0;
    }
    `,
    sgFragmentShaderSource_1sp_fullfloat: `

    varying vec3 vDiffuse;
    varying vec3 vPosition;

    varying vec3 vSgMean0;

    varying float vSgScale0;

    varying vec3 vSgColor0;

    
    uniform mat4 gmatrix_inv;
    uniform mat3 intrinsic;
    uniform mat3 c2w_T;

    uniform float fx;
    uniform float fy;
    uniform float cx;
    uniform float cy;
    uniform mat3 worldspace_R_opengl;


    vec3 evalSphericalGaussian(vec3 direction, vec3 mean, float scale, vec3 color) {
        return color * exp(scale * (dot(direction, mean) - 1.0));
    }

    void main() {


        vec4 positionCamWorld = gmatrix_inv * vec4(vPosition, 1.0);
        positionCamWorld /= positionCamWorld.w;
        vec3 positionScrWorld = intrinsic * vec3(positionCamWorld.xyz);
        positionScrWorld /= positionScrWorld.z;
        
        float px = 2.0 * cx - positionScrWorld.x;
        float py = positionScrWorld.y;
        float _px = (px - cx) / fx;
        float _py = (py - cy) / fy;
        vec3 directionCamWorld = vec3(_px, _py, 1);

        vec3 directionWorld = directionCamWorld * c2w_T;
        
        
        vec3 diffuse = vDiffuse;

        vec3 viewDependence = evalSphericalGaussian(
            directionWorld, (vSgMean0), vSgScale0, vSgColor0);
        
    
        // vec3 color = vec3(positionCamWorld.x, positionCamWorld.y, positionCamWorld.z);
        // vec3 color = vec3(0, px / 1408.0, py / 376.0);
        // vec3 color = (directionWorld + 1.0) / 2.0;
        vec3 color = diffuse + viewDependence;
        
        gl_FragColor = vec4(color, 1.0);
    }
    `,
    sgVertexShaderSource_3sp_fullfloat: `
    attribute vec3 _byte_normal;
    
    attribute vec3 _sg_mean_0;
    attribute vec3 _sg_mean_1;
    attribute vec3 _sg_mean_2;

    attribute float _sg_scale_0;
    attribute float _sg_scale_1;
    attribute float _sg_scale_2;

    attribute vec3 _sg_color_0;
    attribute vec3 _sg_color_1;
    attribute vec3 _sg_color_2;

    varying vec3 vDiffuse;
    varying vec3 vPosition;

    varying vec3 vSgMean0;
    varying vec3 vSgMean1;
    varying vec3 vSgMean2;

    varying float vSgScale0;
    varying float vSgScale1;
    varying float vSgScale2;

    varying vec3 vSgColor0;
    varying vec3 vSgColor1;
    varying vec3 vSgColor2;

    uniform mat3 worldspace_R_opengl;
    
    void main() {
        vec3 positionWorld = position;
        vPosition = position;
        vec4 positionClip = projectionMatrix * modelViewMatrix * vec4(positionWorld, 1.0);
        gl_Position = positionClip;
        positionClip /= positionClip.w;

        vDiffuse = color.rgb;

        vSgMean0 = _sg_mean_0;
        vSgMean1 = _sg_mean_1;
        vSgMean2 = _sg_mean_2;

        vSgScale0 = _sg_scale_0;
        vSgScale1 = _sg_scale_1;
        vSgScale2 = _sg_scale_2;

        vSgColor0 = _sg_color_0;
        vSgColor1 = _sg_color_1;
        vSgColor2 = _sg_color_2;
    }
    `,
    sgFragmentShaderSource_3sp_fullfloat: `

    varying vec3 vDiffuse;
    varying vec3 vPosition;

    varying vec3 vSgMean0;
    varying vec3 vSgMean1;
    varying vec3 vSgMean2;

    varying float vSgScale0;
    varying float vSgScale1;
    varying float vSgScale2;

    varying vec3 vSgColor0;
    varying vec3 vSgColor1;
    varying vec3 vSgColor2;

    uniform mat4 gmatrix_inv;
    uniform mat3 intrinsic;
    uniform mat3 c2w_T;

    uniform float fx;
    uniform float fy;
    uniform float cx;
    uniform float cy;
    uniform mat3 worldspace_R_opengl;


    vec3 evalSphericalGaussian(vec3 direction, vec3 mean, float scale, vec3 color) {
        return color * exp(scale * (dot(direction, mean) - 1.0));
    }

    void main() {
        vec3 diffuse = vDiffuse;
        
        vec4 positionCamWorld = gmatrix_inv * vec4(vPosition, 1.0);
        positionCamWorld /= positionCamWorld.w;
        vec3 positionScrWorld = intrinsic * vec3(positionCamWorld.xyz);
        positionScrWorld /= positionScrWorld.z;
        
        float px = 2.0 * cx - positionScrWorld.x;
        float py = positionScrWorld.y;
        float _px = (px - cx) / fx;
        float _py = (py - cy) / fy;
        vec3 directionCamWorld = vec3(_px, _py, 1);

        vec3 directionWorld = directionCamWorld * c2w_T;

        vec3 viewDependence = evalSphericalGaussian(
            directionWorld, normalize(vSgMean0), vSgScale0, vSgColor0);
        viewDependence += evalSphericalGaussian(
            directionWorld, normalize(vSgMean1), vSgScale1, vSgColor1);
        viewDependence += evalSphericalGaussian(
            directionWorld, normalize(vSgMean2), vSgScale2, vSgColor2);

        vec3 color;
        color = diffuse + viewDependence;
        // color = vec3(1.0, 0.0, 0.0);
        gl_FragColor = vec4(color, 1.0);
    }
    `,
};

