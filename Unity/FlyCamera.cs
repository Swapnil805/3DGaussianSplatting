using UnityEngine;

public class FlyCamera : MonoBehaviour
{    public float moveSpeed = 5f;
    public float lookSpeed = 2f;
    float yaw;
    float pitch;
    void Start()
    {   Cursor.lockState = CursorLockMode.Locked;
        Cursor.visible = false;    }

    void Update()
    {   yaw += lookSpeed * Input.GetAxis("Mouse X");
        pitch -= lookSpeed * Input.GetAxis("Mouse Y");
        pitch = Mathf.Clamp(pitch, -89f, 89f);
        transform.eulerAngles = new Vector3(pitch, yaw, 0f);
        Vector3 dir = new Vector3(
            Input.GetAxis("Horizontal"),
            0,
            Input.GetAxis("Vertical")
        );
        if (Input.GetKey(KeyCode.E)) dir.y += 1;
        if (Input.GetKey(KeyCode.Q)) dir.y -= 1;

        float speed = Input.GetKey(KeyCode.LeftShift) ? moveSpeed * 3f : moveSpeed;
        transform.Translate(dir * speed * Time.deltaTime, Space.Self);
        if (Input.GetKeyDown(KeyCode.Escape))
        {Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;}
    }
}
