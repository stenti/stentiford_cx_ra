#define BUTTON_HOME 2
#define BUTTON_UP 3
#define BUTTON_CIRCLE 4
#define BUTTON_SPIRAL 5

byte lastCircleState = LOW;
byte lastGoState = LOW;
bool goingHome = false;

int count = 0;
float pi = 3.1416;

int microsteps = 16;
float stepsPerRevolution = 200*microsteps;
float spoolCircumference = 100; //mm
float mmPerStep = spoolCircumference/stepsPerRevolution;

float frameX = 970 / mmPerStep; //steps
float frameY = 970 / mmPerStep; //steps
float frameZ = 400 / mmPerStep; //steps

float step_delay = 200 ;//500; // higher = slower

float origin[3] = {frameX/2,frameY/2,0};
float target[3] = {frameX/2,frameY/2,0};

// int motors[4][2] = {{2,3},  //motor 0
//                     {4,5},  //motor 1
//                     {6,7},  //motor 2
//                     {8,9}}; //motor 3

int motors[4][2] = {{6,7},  //motor 0
                    {8,9},  //motor 1
                    {10,11},  //motor 2
                    {12,13}}; //motor 3

float motor_coords[4][3] = {{0,0,frameZ},
                    {frameX,0,frameZ},
                    {frameX,frameY,frameZ},
                    {0,frameY,frameZ}};
                    
int wireLength[4];
int target_wireLength[4];

float getHypotenuse(float A, float B, float C){
  float D = sqrt(pow(A, 2) + pow(B, 2) + pow(C, 2));
  return D;
}

void getWireLength(float pos[]){
  for (int i = 0; i < 4; i++) {
    target_wireLength[i] = (int) round(getHypotenuse(motor_coords[i][0]-pos[0],motor_coords[i][1]-pos[1],motor_coords[i][2]-pos[2]));
    Serial.println(target_wireLength[i]* mmPerStep);
  }
}

float getMin(int a[4]){
  float aux = abs(a[0]);
  for(int x=1; x<=3; x++){
    if(abs(a[x]) < aux){
      aux = abs(a[x]);
    }
  }
  return aux;
}

void moveToTarget(float target[]){
  getWireLength(target);

  int steps[4];
  for(int i=0; i<4; i++){
    steps[i] = target_wireLength[i] - wireLength[i];
  }

  float shortest = getMin(steps);
  float remainder[4] = {0,0,0,0};
  int integer;
  float decimal;

  for(int s=0; s<shortest; s++){
    for(int i=0; i<4; i++){
        integer = int(steps[i] / shortest);
        decimal = (steps[i] / shortest) - integer;
        
        if(abs(remainder[i])>1){
          integer += int(remainder[i]);
          remainder[i]-= int(remainder[i]);
        }
        remainder[i] += decimal;
        move_steps(i,integer);
    }
  }

  for(int i=0; i<4; i++){
    wireLength[i] = target_wireLength[i];
  }
 
}

void move_steps(int motor, int steps){
  if (steps < 0) {
    digitalWrite(motors[motor][0], HIGH);
  } 
  if (steps > 0){
    digitalWrite(motors[motor][0], LOW);
  }
  for (int i = 0; i < abs(steps); i++) {
    // These four lines result in 1 step:
    digitalWrite(motors[motor][1], HIGH);
    delayMicroseconds(step_delay);
    digitalWrite(motors[motor][1], LOW);
    delayMicroseconds(step_delay);
  }
}

float circle(float r, float t){
  target[0] = r*cos(t) + origin[0];
  target[1] = r*sin(t) + origin[1];
  target[2] = origin[2];
}

float spiral(float r, float t){
  target[0] = r*cos(t) + origin[0];
  target[1] = r*sin(t) + origin[1];
  target[2] = (10/mmPerStep)*t + origin[2];
}

float daisy(float r, float t){
  target[0] = r*(sin(2*t)*cos(t)) + origin[0];
  target[1] = r*(sin(2*t)*sin(t)) + origin[1];
  target[2] = (10/mmPerStep)*t + origin[2];
}

// float eight(float r, float t){
//   target[0] = r*sin(2*t) + origin[0];
//   target[1] = r*cos(t) + origin[1];
//   target[2] = origin[2];
// }

float eight(float r, float t){
  target[0] = r*(0.3*sin(2*t)+cos(t)) + origin[0];
  target[1] = r*cos(t) + origin[1];
  target[2] = origin[2];
}

void setup() {
  pinMode(BUTTON_HOME, INPUT);
  pinMode(BUTTON_UP, INPUT);
  pinMode(BUTTON_CIRCLE, INPUT);
  pinMode(BUTTON_SPIRAL, INPUT);
  pinMode(6,OUTPUT);
  pinMode(7,OUTPUT);
  pinMode(8,OUTPUT);
  pinMode(9,OUTPUT);
  pinMode(10,OUTPUT);
  pinMode(11,OUTPUT);
  pinMode(12,OUTPUT);
  pinMode(13,OUTPUT);

  Serial.begin(38400);

  getWireLength(origin);
  for(int i=0; i<4; i++){
    wireLength[i] = 815  / mmPerStep ; //target_wireLength[i];
  }

  attachInterrupt(digitalPinToInterrupt(BUTTON_HOME), gohome, RISING);
}

void loop() {

  byte circleState = digitalRead(BUTTON_CIRCLE);
  byte goState = digitalRead(BUTTON_SPIRAL);
  byte homeState = digitalRead(BUTTON_HOME);

  if (circleState != lastCircleState){
    lastCircleState = circleState;
    if (circleState == HIGH ) {
      for (int i = 0; i <= 360; i=i+9) {
        float t = i*(pi/180);
        circle(150/mmPerStep,t);
        // eight(100/mmPerStep,t);
        moveToTarget(target);
        if (goingHome){
          break;
          goingHome = false;
        }
      }
      moveToTarget(origin);
    }
  }

  while (digitalRead(BUTTON_UP) == HIGH){
    for(int motor = 0; motor < 4; motor++){ 
      digitalWrite(motors[motor][0], HIGH);
      for (int i = 0; i < stepsPerRevolution/microsteps; i++) {
        // These four lines result in 1 step:
          digitalWrite(motors[motor][1], HIGH);
          delayMicroseconds(step_delay);
          digitalWrite(motors[motor][1], LOW);
          delayMicroseconds(step_delay);
      }
    }
  }


if (count == 0){
  getWireLength(origin);
  moveToTarget(origin);
  count++;
}

  if (goState != lastGoState){
    lastGoState = goState;
    if (goState == HIGH ) {
      moveToTarget(origin);
      // ROTATE ON THE SPOT (3 revolutions should be 18000 ms)
      delay(30000);
      // SMALL CIRCLE
      for (int i = 0; i <= 360*3; i=i+3) {
        float t = i*(pi/180);
        circle(50/mmPerStep,t);
        moveToTarget(target);
      }
      delay(1000);
      // moveToTarget(origin);
      // MEDIUM CIRCLE
      for (int i = 0; i <= 360*3; i++) {
        float t = i*(pi/180);
        circle(100/mmPerStep,t);
        moveToTarget(target);
      }
      delay(1000);
      // moveToTarget(origin);
      // LARGE CIRCLE
      for (int i = 0; i <= 360*3; i++) {
        float t = i*(pi/180);
        circle(150/mmPerStep,t);
        moveToTarget(target);
      }
      delay(1000);
      // moveToTarget(origin);
      // VERY LARGE CIRCLE
      for (int i = 0; i <= 360*3; i++) {
        float t = i*(pi/180);
        circle(200/mmPerStep,t);
        moveToTarget(target);
      }
      delay(1000);
      moveToTarget(origin);
    }
  }

}

void gohome(){
  goingHome = true;
  moveToTarget(origin);
}

